# -*- coding: utf-8 -*-

import sys
from os import path, system
from glob import glob
import scipy as sp
import scipy.sparse as smat
from scipy.sparse import identity as speye
import ctypes
from ctypes import *

def genFields(names, types):
	return list(zip(names, types))

def fillprototype(f, restype, argtypes):
	f.restype = restype
	f.argtypes = argtypes

def load_dynamic_library(dirname, soname, forced_rebuild=False):
    try:
        if forced_rebuild:
            system("make -C {} clean lib".format(dirname))
        path_to_so = glob(path.join(dirname, soname) + '*.so')[0]
        _c_lib = CDLL(path_to_so)
    except:
        try :
            system("make -C {} clean lib".format(dirname))
            path_to_so = glob(path.join(dirname, soname) + '*.so')[0]
            _c_lib = CDLL(path_to_so)
        except :
            raise Exception('{soname} library cannot be found and built.'.format(soname=soname))
    return _c_lib

# Wrapper for Scipy/Numpy Matrix
class PyMatrix(ctypes.Structure):
    DENSE_ROWMAJOR = 1
    DENSE_COLMAJOR = 2
    SPARSE = 3
    EYE = 4

    _fields_ = [
            ('rows', c_uint64),
            ('cols', c_uint64),
            ('nnz', c_uint64),
            ('row_ptr', POINTER(c_uint64)),
            ('col_ptr', POINTER(c_uint64)),
            ('row_idx', POINTER(c_uint32)),
            ('col_idx', POINTER(c_uint32)),
            ('val', c_void_p),
            ('val_t', c_void_p),
            ('type', c_int32)
            ]

    def check_identiy(self, A):
        rows, cols = A.shape
        if rows != cols:
            return False
        if isinstance(A, sp.ndarray) and (sp.diag(A) == 1).all() != True:
            return False
        if isinstance(A, smat.spmatrix):
            return smat.csr_matrix(A) - speye(rows).nnz == 0

        return True

    @classmethod
    def identity(cls, size, dtype=sp.float32):
        eye = cls(A=None, dtype=dtype)
        eye.rows = c_uint64(size)
        eye.cols = c_uint64(size)
        eye.nnz = c_uint64(size)
        eye.dtype = dtype
        eye.type = PyMatrix.EYE
        name2type = dict(PyMatrix._fields_)
        for name in ['row_ptr', 'col_ptr', 'row_idx', 'col_idx', 'val', 'val_t']:
            setattr(eye, name,  None)
        return eye

    def __init__(self, A, dtype=sp.float32):
        if A is None:
            return

        self.rows = c_uint64(A.shape[0])
        self.cols = c_uint64(A.shape[1])
        self.py_buf = {}
        self.dtype = dtype
        py_buf = self.py_buf

        if isinstance(A, (smat.csc_matrix, smat.csr_matrix)):
            Acsr = smat.csr_matrix(A)
            Acsc = smat.csc_matrix(A)
            self.type = PyMatrix.SPARSE
            self.nnz = c_uint64(Acsr.indptr[-1])
            py_buf['row_ptr'] = Acsr.indptr.astype(sp.uint64)
            py_buf['col_idx'] = Acsr.indices.astype(sp.uint32)
            py_buf['val_t'] = Acsr.data.astype(dtype)
            py_buf['col_ptr'] = Acsc.indptr.astype(sp.uint64)
            py_buf['row_idx'] = Acsc.indices.astype(sp.uint32)
            py_buf['val'] = Acsc.data.astype(dtype)

        elif isinstance(A, smat.coo_matrix):
            def coo_to_csr(coo):
                nr_rows, nr_cols, nnz, row, col, val = \
                        coo.shape[0], coo.shape[1], coo.data.shape[0], coo.row, coo.col, coo.data
                indptr = sp.cumsum(sp.bincount(row + 1, minlength=(nr_rows + 1)), dtype=sp.uint64)
                indices = sp.zeros(nnz, dtype=sp.uint32)
                data = sp.zeros(nnz, dtype=dtype)
                sorted_idx = sp.argsort(row * nr_cols + col)
                indices[:] = col[sorted_idx]
                data[:] = val[sorted_idx]
                return indptr, indices, data

            def coo_to_csc(coo):
                return coo_to_csr(smat.coo_matrix((coo.data, (coo.col, coo.row)), shape=[coo.shape[1], coo.shape[0]]))

            coo = A.tocoo()
            self.type = PyMatrix.SPARSE
            self.nnz = c_uint64(coo.data.shape[0])
            py_buf['row_ptr'], py_buf['col_idx'], py_buf['val_t'] = coo_to_csr(coo)
            py_buf['col_ptr'], py_buf['row_idx'], py_buf['val'] = coo_to_csc(coo)

        elif isinstance(A, sp.ndarray):
            py_buf['val'] = A.astype(dtype)
            if py_buf['val'].flags.f_contiguous:
                self.type = PyMatrix.DENSE_COLMAJOR
            else:
                self.type = PyMatrix.DENSE_ROWMAJOR
            self.nnz = c_uint64(A.shape[0] * A.shape[1])
        name2type = dict(PyMatrix._fields_)
        for name in py_buf:
            setattr(self, name, py_buf[name].ctypes.data_as(name2type[name]))

def svm_read_problem(data_file_name, return_scipy=True):
    """
    svm_read_problem(data_file_name, return_scipy=False) -> [y, x], y: list, x: list of dictionary
    svm_read_problem(data_file_name, return_scipy=True)  -> [y, x], y: ndarray, x: csr_matrix

    Read LIBSVM-format data from data_file_name and return labels y
    and data instances x.
    """
    scipy = sp
    prob_y = []
    prob_x = []
    row_ptr = [0]
    col_idx = []
    for i, line in enumerate(open(data_file_name)):
        line = line.split(None, 1)
        # In case an instance with all zero features
        if len(line) == 1: line += ['']
        label, features = line
        prob_y += [float(label)]
        if scipy != None and return_scipy:
            nz = 0
            for e in features.split():
                ind, val = e.split(":")
                val = float(val)
                if val != 0:
                    col_idx += [int(ind)-1]
                    prob_x += [val]
                    nz += 1
            row_ptr += [row_ptr[-1]+nz]
        else:
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            prob_x += [xi]
    if scipy != None and return_scipy:
        prob_y = scipy.array(prob_y)
        prob_x = scipy.array(prob_x)
        col_idx = scipy.array(col_idx)
        row_ptr = scipy.array(row_ptr)
        prob_x = smat.csr_matrix((prob_x, col_idx, row_ptr))
    return (prob_y, prob_x)
