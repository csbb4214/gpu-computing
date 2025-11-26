CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -g3 -O3
DEPS=
LIBS=-lm -lgomp

FLAGS=
SUFFIX=
ifdef N
FLAGS+=-DN=$(N)
SUFFIX:=_N$(N)
endif

all: matrix_mul$(SUFFIX) matrix_mul_omp$(SUFFIX)

matrix_mul_omp$(SUFFIX): matrix_mul_omp.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS) $(LIBS) $(FLAGS) -fopenmp

matrix_mul$(SUFFIX): matrix_mul.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS) $(LIBS) $(FLAGS)

clean:
	find . -executable -type f -regextype posix-egrep -regex "./matrix_mul(_omp)?(_N[0-9]+)?" -delete

.PHONY: all clean
