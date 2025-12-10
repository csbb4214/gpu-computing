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
ifdef FLOAT
FLAGS+=-DFLOAT
SUFFIX:=$(SUFFIX)_float
endif

ifeq ($(OS),Windows_NT)
CLEAN=del /Q scan*.exe
else
CLEAN=find . -executable -type f -regextype posix-egrep -regex "./(scan)?(_N[0-9]+)?(_float)?" -delete
endif

all: scan$(SUFFIX)

scan$(SUFFIX): scan.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(LDFLAGS) $(LIBS) $(FLAGS)

clean:
	$(CLEAN)

.PHONY: all clean