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

# matrix_mul.c uses USE_DOUBLE (not FLOAT), so we handle that here
ifdef USE_DOUBLE
FLAGS+=-DUSE_DOUBLE
SUFFIX:=$(SUFFIX)_double
else
SUFFIX:=$(SUFFIX)_float
endif

ifeq ($(OS),Windows_NT)
CLEAN=del /Q matrix_mul*.exe
else
CLEAN=find . -executable -type f -regextype posix-egrep -regex "./matrix_mul(_N[0-9]+)?(_float|_double)?" -delete
endif

all: matrix_mul$(SUFFIX)

matrix_mul$(SUFFIX): matrix_mul.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS)

clean:
	$(CLEAN)

.PHONY: all clean

