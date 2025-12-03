CC=gcc
CFLAGS=-std=c17 -I. -Wall -Wextra -Wpedantic -DCL_TARGET_OPENCL_VERSION=300 -g3 -O3
LDFLAGS=
DEPS=clu_errcheck.h clu_setup.h stb_image.h stb_image_write.h
SRC=clu_setup.c
LIBS=-lOpenCL -lm -lgomp

TARGETS=auto_levels auto_levels_cl

ifeq ($(OS),Windows_NT)
    CLEAN=del /Q auto_levels*.exe
    RM=del /Q
else
    CLEAN=rm -f auto_levels auto_levels_cl
    RM=rm -f
endif

all: $(TARGETS)

# Serial
auto_levels: auto_levels.c $(DEPS)
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS) -lgomp

# OpenCL
auto_levels_cl: auto_levels_cl.c $(SRC) $(DEPS)
	$(CC) -o $@ $< $(SRC) $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	$(CLEAN)

# Run using test.png
test: test-serial test-ocl

test-serial:
	./auto_levels test.png test_output_serial.png

test-ocl:
	./auto_levels_cl test.png test_output_ocl.png

# Run using earth-huge.png
run: run-serial run-ocl

run-serial:
	./auto_levels earth-huge.png earth-huge_output_serial.png

run-ocl:
	./auto_levels_cl earth-huge.png earth-huge_output_ocl.png

.PHONY: all clean test run