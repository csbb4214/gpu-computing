CC = gcc

OPENCL_LIB = -lOpenCL

CFLAGS = -Wall -Wextra -O2 -std=c11 -fopenmp
INCLUDES = -I.

TARGET = list_sort_ocl

SRCS = list_sort_ocl.c clu_setup.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(OPENCL_LIB)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
