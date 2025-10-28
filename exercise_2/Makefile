CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300 -g3 -O3
LDFLAGS=-L"C:\OpenCL-SDK\lib"
DEPS=clu_errcheck.h
LIBS=-lOpenCL -lm -lgomp

FLAGS=
SUFFIX=
ifdef N
FLAGS+=-DN=$(N)
SUFFIX:=_N$(N)
endif
ifdef IT
FLAGS+=-DIT=$(IT)
SUFFIX:=$(SUFFIX)_IT$(IT)
endif

all: jacobi$(SUFFIX) jacobi_omp$(SUFFIX) jacobi_ocl$(SUFFIX)

jacobi_omp$(SUFFIX): jacobi.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS) -fopenmp

jacobi$(SUFFIX): jacobi.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS)

jacobi_ocl$(SUFFIX): jacobi_ocl.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS)

clean:
	find . -executable -type f -regextype posix-egrep -regex "./jacobi(_omp|_ocl)?(_N[0-9]+)?(_IT[0-9]+)?" -delete

.PHONY: all clean
