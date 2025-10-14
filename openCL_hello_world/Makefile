CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300
DEPS=clu_errcheck.h
LIBS=-lOpenCL

all: hello info

hello: hello.c $(DEPS)
info: info.c $(DEPS)

% : %.c
	$(CC) -o $@ $< $(CFLAGS) $(LIBS)

clean:
	rm -f hello info

.PHONY: all clean
