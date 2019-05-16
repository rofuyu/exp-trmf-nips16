import sys, os
from os import path, system
import collections
import itertools
import pickle

import scipy as sp
import scipy.sparse as smat

try:
    from .rf_util import PyMatrix, fillprototype, load_dynamic_library
except:
    from rf_util import PyMatrix, fillprototype, load_dynamic_library

import ctypes
from ctypes import *


class corelib(object):
    def __init__(self, dirname, soname, forced_rebuild=False):
        self.clib_float32 = load_dynamic_library(dirname, soname + '_float32', forced_rebuild=forced_rebuild)
        self.clib_float64 = load_dynamic_library(dirname, soname + '_float64', forced_rebuild=forced_rebuild)
        arg_list = [
                    POINTER(PyMatrix),  # PyMatrix Y
                    POINTER(c_uint32),  # py_lag_set
                    c_uint32,           # py_lag_size
                    POINTER(PyMatrix),  # PyMatrix W
                    POINTER(PyMatrix),  # PyMatrix H
                    POINTER(PyMatrix),  # PyMatrix lag_val
                    c_int32,            # warm_start
                    c_double,           # lambdaI
                    c_double,           # lambdaAR
                    c_double,           # lambdaLag
                    c_int32,            # max_iter
                    c_int32,            # period_W
                    c_int32,            # period_H
                    c_int32,            # period_Lag
                    c_int32,            # threads
                    c_int32,            # missing
                    c_int32,            # verbose
                ]
        fillprototype(self.clib_float32.c_trmf_train, None, arg_list)
        fillprototype(self.clib_float64.c_trmf_train, None, arg_list)

    def train(self, pyY, lag_set, pyW, pyH, pylag_val, warm_start=True,
            lambdaI=0.1, lambdaAR=0.1, lambdaLag=0.1, max_iter=10,
            period_W=1, period_H=1, period_Lag=2, threads=1, missing=False, verbose=0):
        clib = self.clib_float32
        if pyY.dtype == sp.float64:
            clib = self.clib_float64
            if verbose != 0:
                print('perform float64 computation')
        else:
            clib = self.clib_float32
            if verbose != 0:
                print('perform float32 computation')
        clib.c_trmf_train(
                byref(pyY),
                lag_set.ctypes.data_as(POINTER(c_uint32)),
                len(lag_set),
                byref(pyW),
                byref(pyH),
                byref(pylag_val),
                c_int32(warm_start),
                lambdaI,
                lambdaAR,
                lambdaLag,
                max_iter,
                period_W,
                period_H,
                period_Lag,
                threads,
                missing,
                verbose
            )

forced_rebuild = False
corelib_path = path.join(path.dirname(path.abspath(__file__)), 'corelib/')
soname = 'trmf'
_clib = corelib(corelib_path, soname, forced_rebuild=forced_rebuild)

class NormalizedTransform(object):
    def __init__(self, Y):
        mean = sp.array(sp.mean(Y, axis=0)).reshape(1, -1)
        std = sp.array(sp.std(Y, axis=0)).reshape(1, -1)
        std[std==0] = 1.0
        self.a = 1. / std
        self.b = - self.a * mean

    def preprocess(self, Y):
        assert Y.shape[1] == self.a.shape[1]
        return Y * self.a + self.b

    def postprocess(self, Y):
        assert Y.shape[1] == self.a.shape[1]
        return (Y - self.b) / self.a

class Model(object):
    def __init__(self, pyW=None, pyH=None, pylag_val=None, lag_set=None, transform=None):
        self.pyW = pyW
        self.pyH = pyH
        self.pylag_val = pylag_val
        self.lag_set = lag_set
        self.transform = transform

    @property
    def k(self):
        return self.W.shape[1]

    @property
    def m(self):
        return self.W.shape[0]

    @property
    def n(self):
        return self.H.shape[0]

    @property
    def W(self):
        return self.pyW.py_buf['val']

    @property
    def H(self):
        return self.pyH.py_buf['val']

    @property
    def lag_val(self):
        return self.pylag_val.py_buf['val']


    @classmethod
    def load(cls, path_to_folder, dtype=None):
        assert path.isdir(path_to_folder)
        other_members = path.join(path_to_folder, 'other.pkl')
        with open(other_members, 'rb') as other:
            other = pickle.load(other)
            transform = other['transform']
        np_arrays = path.join(path_to_folder, 'arrays.npz')
        with open(np_arrays, 'rb') as npz:
            npz = sp.load(npz)
            W = npz['W']
            H = npz['H']
            lag_val = npz['lag_val']
            lag_set = npz['lag_set']
        if dtype is None:
            dtype = W.dtype
        return cls(pyW=PyMatrix(W, dtype),
                   pyH=PyMatrix(H, dtype),
                   pylag_val=PyMatrix(lag_val, dtype),
                   lag_set=lag_set,
                   transform=transform
                  )

    def save(self, path_to_folder):
        if not path.exists(path_to_folder):
            os.makedirs(path_to_folder)
        else:
            assert path.isdir(path_to_folder)
        np_arrays = path.join(path_to_folder, 'arrays.npz')
        with open(np_arrays, 'wb') as npz:
            sp.savez(npz, W=self.W,
                          H=self.H,
                          lag_val=self.lag_val,
                          lag_set=self.lag_set)
        other_members = path.join(path_to_folder, 'other.pkl')
        with open(other_members, 'wb') as other:
            tmp = {'transform': self.transform}
            pickle.dump(tmp, other)

    def latent_forecast(self, window, Wnew=None):
        if Wnew is not None:
            assert Wnew.shape[0] == self.m + window
            assert Wnew.shape[1] == self.k
            assert Wnew.dtype == self.W.dtype
            assert Wnew.flags['C_CONTIGUOUS'] == True
        else:
            Wnew = sp.zeros((self.m + window, self.k), dtype=self.W.dtype, order='C')
        Wnew[:self.m, :] = self.W[:]
        for i in range(self.m, self.m + window):
            Wnew[i, :] = (Wnew[i - self.lag_set, :] * self.lag_val).sum(axis=0)
        return Wnew

    def forecast(self, window, Ynew=None, threshold=None):
        Wnew = self.latent_forecast(window)
        Wnew = Wnew[self.m:, :]
        if Ynew is None:
            Ynew = sp.zeros((window, self.n), dtype=self.W.dtype, order='C')
        sp.dot(Wnew, self.H.T, Ynew)
        if threshold is not None:
            Ynew[Ynew < threshold] = threshold
        if self.transform is not None:
            Ynew[:] = self.transform.postprocess(Ynew)[:]
        return Ynew, Wnew

    @staticmethod
    def syn_gen(m, n, k, lag_set, seed=None, noise=0.01, dtype=sp.float32):
        if seed is not None:
            sp.random.seed(seed)
        lag_set = sp.array(sorted(lag_set), dtype=sp.uint32)
        lag_size = len(lag_set)
        midx = lag_set.max()

        W = sp.zeros((m, k), dtype=dtype, order='C')
        H = sp.zeros((n, k), dtype=dtype, order='C')
        lag_val = sp.zeros((lag_size, k), dtype=dtype, order='F')
        Y = sp.zeros((m, n), dtype=dtype, order='C')

        W[:] = sp.randn(*W.shape)
        H[:] = sp.randn(*H.shape)
        lag_val[:] = sp.randn(*lag_val.shape)

        lag_val = sp.dot(lag_val, sp.diag(1. / (sp.absolute(lag_val).sum(axis=0) + 0.1)))

        for i in range(midx, m):
            W[i, :] = (W[i - lag_set, :] * lag_val).sum(axis=0)
        W[midx:, :] += noise * sp.randn(m - midx, k)
        sp.dot(W, H.T, Y)
        #Y += noise * sp.randn(*Y.shape)
        data = {'W':W, 'H':H, 'lag_val':lag_val, 'lag_set':lag_set, 'Y':Y, 'k':k}
        return data

    @classmethod
    def initialize(cls, Y, lag_set, k, warm_start_model=None, seed=None, dtype=None, transform=None):
        if seed is not None:
            sp.random.seed(seed)
        if dtype is None:
            dtype = Y.dtype
        m, n = Y.shape[0], Y.shape[1]
        lag_set = sp.array(sorted(lag_set), dtype=sp.uint32)
        lag_size = len(lag_set)
        W = sp.zeros((m, k), dtype=dtype, order='C')
        H = sp.zeros((n, k), dtype=dtype, order='C')
        lag_val = sp.zeros((lag_size, k), dtype=dtype, order='F')
        W[:] = sp.rand(*W.shape)
        H[:] = sp.rand(*H.shape)
        lag_val[:] = sp.randn(*lag_val.shape)
        if warm_start_model is not None:
            assert warm_start_model.k == k
            assert warm_start_model.n == n
            assert warm_start_model.m <= m
            assert len(lag_set) == len(warm_start_model.lag_set)
            W[:] = warm_start_model.latent_forecast(m - warm_start_model.m)
            #W[:warm_start_model.m, :] = warm_start_model.W[:, :]
            H[:] = warm_start_model.H[:]
            lag_val[:] = warm_start_model.lag_val[:]
            transform = warm_start_model.transform
        if transform is not None:
            transform = NormalizedTransform(Y)

        return cls(pyW=PyMatrix(W, dtype), pyH=PyMatrix(H, dtype), pylag_val=PyMatrix(lag_val, dtype),
                    lag_set=lag_set, transform=transform)

def train(Y, model, lambdaI=0.1, lambdaAR=0.1, lambdaLag=0.1,
        max_iter=10, period_W=1, period_H=1, period_Lag=2,
        threads=1, missing=False, verbose=0):
    if model.transform is not None:
        Y = model.transform.preprocess(Y)
    _clib.train(PyMatrix(Y, dtype=model.W.dtype), model.lag_set,
            model.pyW, model.pyH, model.pylag_val, warm_start=True,
            lambdaI=lambdaI, lambdaAR=lambdaAR, lambdaLag=lambdaLag,
            max_iter=max_iter, period_W=period_W, period_H=period_H, period_Lag=period_Lag,
            threads=threads, missing=missing, verbose=verbose)

    return model

class Metrics(collections.namedtuple('Metrics', ['nd', 'mase', 'nrmse', 'm_nd', 'm_mase', 'm_nrmse', 'mape'])):
    __slots__ = ()

    def __str__(self):
        return  ' '.join("{}={:.4g}".format(key, getattr(self, key)) for key in self._fields)

    @classmethod
    def default(cls):
        return cls(nd=1e10, mase=1e10, nrmse=1e10, m_nd=1e10, m_mase=1e10, m_nrmse=1e10, mape=1e10)

    @classmethod
    def generate(cls, trueY, forecastY, missing=True):
        nz_mask = trueY != 0
        diff = forecastY - trueY
        abs_true = sp.absolute(trueY)
        abs_diff = sp.absolute(diff)

        def my_mean(x):
            tmp = x[sp.isfinite(x)]
            assert len(tmp) != 0
            return tmp.mean()

        with sp.errstate(divide='ignore'):
            nrmse = sp.sqrt((diff**2).mean()) / abs_true.mean()
            m_nrmse = my_mean(sp.sqrt((diff**2).mean(axis=0)) / abs_true.mean(axis=0))

            nd = abs_diff.sum() / abs_true.sum()
            m_nd = my_mean(abs_diff.sum(axis=0) / abs_true.sum(axis=0))

            abs_baseline = sp.absolute(trueY[1:,:] - trueY[:-1,:])
            mase = abs_diff.mean() / abs_baseline.mean()
            m_mase = my_mean(abs_diff.mean(axis=0) / abs_baseline.mean(axis=0))

            mape = my_mean(sp.divide(abs_diff, abs_true, where=nz_mask))

        return cls(nd=nd, mase=mase, nrmse=nrmse, m_nd=m_nd, m_mase=m_mase, m_nrmse=m_nrmse, mape=mape)

def rolling_validate(Y, lag_set, k=40, window_size=24, nr_windows=7, lambdaI=0.5, lambdaAR=50, lambdaLag=0.5,
        max_iter=20, missing=True, threshold=0, transform=None, threads=16, verbose=0, seed=0):
    T = Y.shape[0]  # length of time series
    n = Y.shape[1]  # dimension of time series

    assert T > nr_windows * window_size
    trueY = Y[-(nr_windows * window_size):, :]
    forecastY = sp.zeros((nr_windows * window_size, n), dtype=Y.dtype, order='C')

    # metrics
    prev_model = None
    for i in range(nr_windows):
        trn_start, trn_end = 0, T - (nr_windows - i) * window_size
        trn_idx = slice(trn_start, trn_end)
        tst_idx = slice(trn_end, trn_end + window_size)
        fct_idx = slice(i * window_size, (i + 1) * window_size)
        Y_trn = Y[trn_idx, :]
        if missing:
            Y_trn = smat.csr_matrix(Y_trn)
        curr_model = Model.initialize(Y_trn, lag_set, k, seed=seed, warm_start_model=prev_model, transform=transform)
        curr_model = train(Y_trn, curr_model,
                lambdaI=lambdaI, lambdaAR=lambdaAR, lambdaLag=lambdaLag,
                max_iter=max_iter, missing=missing, threads=threads, verbose=verbose)
        curr_model.forecast(window_size, Ynew=forecastY[fct_idx, :], threshold=threshold)
        prev_model = curr_model

    return Metrics.generate(trueY, forecastY, missing=missing)

def grid_search(Y, lag_set, grid_params, pkl_file=None, **kw_args):
    results = []
    best = Metrics.default()

    keys = list(grid_params.keys())
    for values in itertools.product(*[grid_params[k] for k in keys]):
        new_kw_args = kw_args.copy()
        new_kw_args.update(dict(zip(keys, values)))
        metrics = rolling_validate(Y, lag_set, **new_kw_args)
        results += [{'kws': new_kw_args, 'metrics': metrics}]
        if metrics.m_nd < best.m_nd:
            best = metrics
            print(metrics, dict(zip(keys, values)))
        if pkl_file is not None:
            pickle.dump(results, open(pkl_file, 'wb'))
    return results, best

if __name__ == '__main__':
    def norm(x):
        return (x * x).sum()

    missing = True
    missing = False
    m, n, k, lag_set = 1000, 500, 20, list(range(24)) + list(range(24 * 7, 24 * 8))
    dtype = sp.float64
    threads = 16
    data = Model.syn_gen(m, n, k, lag_set, seed=0, dtype=dtype)
    data['Y'] += 10
    m0 = Model.initialize(data['Y'], data['lag_set'], k + 20, seed=0)
    #m0 = Model.initialize(data['Y'], data['lag_set'], k, seed=0)
    print('dY={} W{} H{}'.format(norm(data['Y'] - m0.W.dot(m0.H.T)), norm(m0.W), norm(m0.H)))
    train(data['Y'], m0, lambdaI=0.01, lambdaAR=0.001, lambdaLag=0.0001, max_iter=20, missing=missing, verbose=1, threads=threads)
    print('dY={} W{} H{}'.format(norm(data['Y'] - m0.W.dot(m0.H.T)), norm(m0.W), norm(m0.H)))

    print(rolling_validate(data['Y'], data['lag_set'], k, 24, 7, lambdaI=0.001, lambdaAR=0.001, lambdaLag=0.2, threshold=None))

