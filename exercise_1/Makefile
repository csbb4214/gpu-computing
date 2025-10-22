CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300
LIBS=-lOpenCL

all: matrix_mul

matrix_mul: matrix_mul.c

% : %.c
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

clean:
	rm -f matrix_mul

.PHONY: all clean
