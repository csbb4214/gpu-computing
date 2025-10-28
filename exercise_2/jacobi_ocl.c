#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clu_errcheck.h"

#define MEM_SIZE 128

static char* load_kernel_source(const char* path, size_t* out_size) {
	FILE* file = fopen(path, "rb");
	if(file == NULL) {
		fprintf(stderr, "Failed to open kernel source '%s': %s\n", path, strerror(errno));
		return NULL;
	}

	if(fseek(file, 0L, SEEK_END) != 0) {
		fprintf(stderr, "Failed to seek kernel source '%s'.\n", path);
		fclose(file);
		return NULL;
	}

	long length = ftell(file);
	if(length < 0) {
		fprintf(stderr, "Failed to determine size of kernel source '%s'.\n", path);
		fclose(file);
		return NULL;
	}
	rewind(file);

	char* buffer = (char*)malloc((size_t)length + 1U);
	if(buffer == NULL) {
		fprintf(stderr, "Out of memory reading kernel source '%s'.\n", path);
		fclose(file);
		return NULL;
	}

	const size_t read = fread(buffer, 1U, (size_t)length, file);
	if(read != (size_t)length) {
		fprintf(stderr, "Failed to read kernel source '%s'.\n", path);
		free(buffer);
		fclose(file);
		return NULL;
	}

	buffer[length] = '\0';
	if(out_size != NULL) { *out_size = read; }

	fclose(file);
	return buffer;
}

int main(void) {
	const char kernel_path[] = "./jacobi.cl";
	size_t source_size = 0;
	char* source_str = load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) { return EXIT_FAILURE; }

	cl_platform_id platform_id = NULL;
	cl_uint platform_count = 0;
	CLU_ERRCHECK(clGetPlatformIDs(1, &platform_id, &platform_count));

	cl_device_id device_id = NULL;
	cl_uint device_count = 0;
	CLU_ERRCHECK(clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &device_count));

	cl_int err = CL_SUCCESS;
	cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	CLU_ERRCHECK(err);

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, NULL, &err);
	CLU_ERRCHECK(err);

	cl_mem memobj = clCreateBuffer(context, CL_MEM_READ_WRITE, MEM_SIZE, NULL, &err);
	CLU_ERRCHECK(err);

	const char* sources[] = {source_str};
	const size_t lengths[] = {source_size};
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
	CLU_ERRCHECK(err);

	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(err != CL_SUCCESS) { cluPrintProgramBuildLog(program, device_id); }
	CLU_ERRCHECK(err);

	cl_kernel kernel = clCreateKernel(program, "jacobi", &err);
	CLU_ERRCHECK(err);

	CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memobj));

	const size_t global_work_size[] = {1U};
	CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, global_work_size, NULL, 0, NULL, NULL));

	char string[MEM_SIZE] = {0};
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0, sizeof(string), string, 0, NULL, NULL));

	puts(string);

	CLU_ERRCHECK(clFlush(command_queue));
	CLU_ERRCHECK(clFinish(command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(memobj));
	CLU_ERRCHECK(clReleaseCommandQueue(command_queue));
	CLU_ERRCHECK(clReleaseContext(context));

	free(source_str);

	return EXIT_SUCCESS;
}
