CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -g -fPIC -fopenmp -std=gnu++0x
CFLAGS = -Wall -Wconversion -O2 -fPIC -fopenmp -std=gnu++0x
BLASFLAGS = -llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib
BLASFLAGS = -lblas -llapack

# ======================== flags for ICC on stampede ===========================
#CXX = icc
#CC = icc
#ICC_CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=gnu++0x
#ICC_BLAS = -I$MKLROOT/include -mkl=parallel -openmp 
#BLASFLAGS = $(ICC_BLAS)
#CFLAGS = $(ICC_CFLAGS)

# ======================== flags for G++ on stampede ===========================
# module swap intel gcc; module load mkl 
#CXX = g++ 
#CC = gcc
#GCC_CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=gnu++0x
#GCC_BLAS = -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
#BLASFLAGS = ${GCC_BLAS}
#CFLAGS = ${GCC_CFLAGS}


LIBSFLAGS= $(BLASFLAGS) -lz -Lzlib 

LIBS = blas/blas.a
SHVER = 1
OS = $(shell uname)
#LIBS = -lblas

ALLLIB = dbilinear.o imf.o smat.o dmat.o tron.o blas/blas.a zlib_util.o zlib/libz.a 

#all: train-ml  test-smat test-blas
#all: train-ml train-mf train-mf-test
#all: train-ml train-mf train-mf-test
all: lib 
lib: $(ALLLIB)

tar: 
	tar cvzf imf-new.tgz *.cpp *.h Makefile blas zlib matlab python

dbilinear.o: dbilinear.cpp dbilinear.h tron.h smat.h
	$(CXX) $(CFLAGS) -c -o dbilinear.o dbilinear.cpp 

imf.o: imf.cpp imf.h dbilinear.h smat.h dmat.h
	$(CXX) $(CFLAGS) -c -o imf.o imf.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

smat.o: smat.h smat.cpp 
	${CXX} ${CFLAGS} -c -o smat.o smat.cpp

dmat.o: dmat.h dmat.cpp 
	${CXX} ${CFLAGS} -c -o dmat.o dmat.cpp

zlib_util.o: zlib_util.h zlib_util.cpp
	${CXX} ${CFLAGS} -c -o zlib_util.o zlib_util.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

zlib/libz.a: 
	make -C zlib

clean:
	rm -f *~ *.o 
	make -C blas clean
	make -C zlib clean
	make -C matlab clean
	make -C octave clean
