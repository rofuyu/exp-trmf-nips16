CXX?=g++
CXXFLAGS=-fopenmp -march=native -O3 -std=c++11 -fPIC

SONAME=trmf

#BLASFLAGS?=-lblas -llapack
#BLASFLAGS?=-lmklml_intel
#BLASLIB=/home/rofuyu/anaconda3/lib/python3.7/site-packages/numpy/core/../../../../libmkl_rt.so
#LIBSFLAGS=${BLASLIB} -Wl,-rpath,$(basename ${BLASLIB})
#LIBSFLAGS=${BLASLIB}

SCIPY_BLAS=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print(' '.join('-l{}'.format(x) for x in C.get_info('lapack_opt')['libraries']))" | tail -n 1)
SCIPY_BLAS_DIR=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print(' '.join('-L{}'.format(x) for x in C.get_info('lapack_opt')['library_dirs']))" | tail -n 1)
SCIPY_BLAS_RPATH=$(shell python -W ignore -c "import numpy.distutils.system_info as C;print('-Wl,-rpath,{}'.format(':'.join(C.get_info('lapack_opt')['library_dirs'])))" | tail -n 1)
LIBSFLAGS=${SCIPY_BLAS_RPATH} ${SCIPY_BLAS_DIR} ${SCIPY_BLAS}

OS=$(shell uname)

all: trmf.o
	${CXX} ${CXXFLAGS} -Wall -c -o trmf.o trmf.cpp

lib: trmf.cpp trmf.h
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,${SONAME}"; \
		BLAS_LIB_FLAG="-framework Accelerate";\
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,${SONAME}"; \
		BLAS_LIB_FLAG=" -Wl,--as-needed ${SCIPY_BLAS_RPATH} ${SCIPY_BLAS_DIR} ${SCIPY_BLAS} -liomp5";\
	fi; \
	echo CXXFLAGs=${CXXFLAGS}; \
	echo BLAS_LIB_FLAG=$${BLAS_LIB_FLAG}; \
	${CXX} ${CXXFLAGS} -D ValueType=float $${SHARED_LIB_FLAG} -o ${SONAME}_float32.so trmf.cpp $${BLAS_LIB_FLAG} ; \
	${CXX} ${CXXFLAGS} -D ValueType=double $${SHARED_LIB_FLAG} -o ${SONAME}_float64.so trmf.cpp $${BLAS_LIB_FLAG}; \

clean:
	rm -rf ${SONAME} ${SONAME}*.so *.o
