CC=gcc
CFLAGS=-std=c11 -I. -Wall -g3 -O3 -fopenmp
DEPS=stb_image.h stb_image_write.h
LIBS=-lOpenCL -lm

all: auto_levels

auto_levels: auto_levels.c
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS) $(FLAGS)

clean:
	rm auto_levels

.PHONY: all clean
