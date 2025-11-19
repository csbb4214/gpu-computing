CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300 -g3 -O3
LDFLAGS=
DEPS=clu_errcheck.h clu_setup.h
SRC=clu_setup.c
LIBS=-lOpenCL -lm -lgomp

FLAGS=
SUFFIX=
ifdef N
FLAGS+=-DN=$(N)
SUFFIX:=_N$(N)
endif
ifdef LOCAL_WORKGROUP_SIZE
FLAGS+=-DLOCAL_WORKGROUP_SIZE=$(LOCAL_WORKGROUP_SIZE)
SUFFIX:=$(SUFFIX)_WG$(LOCAL_WORKGROUP_SIZE)
endif
ifdef FLOAT
FLAGS+=-DFLOAT
SUFFIX:=$(SUFFIX)_float
endif

ifeq ($(OS),Windows_NT)
CLEAN=del /Q sequential_reduction*.exe parallel_reduction*.exe
else
CLEAN=find . -executable -type f -regextype posix-egrep -regex "./(sequential_reduction|parallel_reduction)?(_N[0-9]+)?(_WG[0-9]+)?(_float)?" -delete
endif

all: sequential_reduction$(SUFFIX) parallel_reduction$(SUFFIX)

sequential_reduction$(SUFFIX): sequential_reduction.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(FLAGS) $(LIBS)

parallel_reduction$(SUFFIX): parallel_reduction.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS)

clean:
	$(CLEAN)

.PHONY: all clean
