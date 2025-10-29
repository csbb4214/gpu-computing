#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clu_errcheck.h"

#ifndef N
#define N 1024
#endif
#ifndef IT
#define IT 100
#endif
#ifdef FLOAT
#define VALUE float
#define KERNEL_NAME "jacobi_step_float"
#else
#define VALUE double
#define KERNEL_NAME "jacobi_step_double"
#endif
#ifndef VERSION
#define VERSION 1
#endif

VALUE u[N][N], tmp[N][N], f[N][N];

VALUE init_func(int x, int y) {
	return 40 * sin((VALUE)(16 * (2 * x - 1) * y));
}

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

	// init matrix
	memset(u, 0, sizeof(u));

	// init F
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			f[i][j] = init_func(i, j);
		}
	}

	const VALUE factor = pow((VALUE)1 / N, 2);

	const size_t bytes = sizeof(VALUE) * N * N;

	cl_mem buf_u = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_f = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	CLU_ERRCHECK(clEnqueueWriteBuffer(command_queue, buf_f, CL_TRUE, 0, bytes, f, 0, NULL, NULL));

	const char* sources[] = {source_str};
	const size_t lengths[] = {source_size};
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
	CLU_ERRCHECK(err);

	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	if(err != CL_SUCCESS) { cluPrintProgramBuildLog(program, device_id); }
	CLU_ERRCHECK(err);

	cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);


	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_f));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(VALUE), (void*)&factor));

	const double start_time = omp_get_wtime();

	const size_t global_work_size[2] = {(size_t)N, (size_t)N};
#if VERSION == 1
	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clEnqueueWriteBuffer(command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));

		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_tmp));

		CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL));

		CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, buf_tmp, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
	}
#else
	CLU_ERRCHECK(clEnqueueWriteBuffer(command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_tmp));

		CLU_ERRCHECK(clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL));
		cl_mem temp = buf_u;
		buf_u = buf_tmp;
		buf_tmp = temp;
	}
	CLU_ERRCHECK(clEnqueueReadBuffer(command_queue, buf_tmp, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
#endif

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

	// Calculate checksum
	VALUE checksum = 0;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			checksum += u[i][j];
		}
	}

#ifdef FLOAT
	const char* prec = "float";
#else
	const char* prec = "double";
#endif

#if VERSION == 1
	printf("opencl_V1,%s,%d,%d,%.3f,%.15e\n", prec, N, IT, elapsed_ms, checksum);
#else
	printf("opencl_V2,%s,%d,%d,%.3f,%.15e\n", prec, N, IT, elapsed_ms, checksum);
#endif

	CLU_ERRCHECK(clFlush(command_queue));
	CLU_ERRCHECK(clFinish(command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(buf_u));
	CLU_ERRCHECK(clReleaseMemObject(buf_tmp));
	CLU_ERRCHECK(clReleaseMemObject(buf_f));
	CLU_ERRCHECK(clReleaseCommandQueue(command_queue));
	CLU_ERRCHECK(clReleaseContext(context));

	free(source_str);

	return EXIT_SUCCESS;
}
