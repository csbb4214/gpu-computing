CC = gcc
CFLAGS = -std=c17 -O3 -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300 -I.
LIBS = -lOpenCL -lgomp

# Matrixgrößen
NS = 512 1024 2000 2048

# Binaries
FLOAT_BINS  = $(foreach N,$(NS),matrix_mul_N$(N)_float)
DOUBLE_BINS = $(foreach N,$(NS),matrix_mul_N$(N)_double)

# Source files
SRC = matrix_mul.c clu_setup.c

all: $(FLOAT_BINS) $(DOUBLE_BINS)

# Float
matrix_mul_N%_float: $(SRC) clu_errcheck.h clu_setup.h
	$(CC) $(SRC) -o $@ $(CFLAGS) $(LIBS) -DN=$*

# Double
matrix_mul_N%_double: $(SRC) clu_errcheck.h clu_setup.h
	$(CC) $(SRC) -o $@ $(CFLAGS) $(LIBS) -DUSE_DOUBLE -DN=$*

clean:
	rm -f matrix_mul_N*_*
	rm -f matrix_mul_*.err matrix_mul_*.out
	rm -f matrix_mul_results_*.csv

.PHONY: all clean

