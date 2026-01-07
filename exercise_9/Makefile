CC = gcc

CFLAGS = -Wall -Wextra -O2 -std=c11 -fopenmp
INCLUDES = -I.

OPENCL_LIB = -lOpenCL

TARGET_CPU = list_sort
TARGET_OCL = list_sort_ocl

CPU_SRCS = list_sort.c
OCL_SRCS = list_sort_ocl.c clu_setup.c    # <- clu_errcheck.c entfernt

CPU_OBJS = $(CPU_SRCS:.c=.o)
OCL_OBJS = $(OCL_SRCS:.c=.o)

all: $(TARGET_CPU) $(TARGET_OCL)

$(TARGET_CPU): $(CPU_OBJS)
	$(CC) $(CFLAGS) -o $@ $^

$(TARGET_OCL): $(OCL_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(OPENCL_LIB)

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(CPU_OBJS) $(OCL_OBJS) $(TARGET_CPU) $(TARGET_OCL)

.PHONY: all clean

