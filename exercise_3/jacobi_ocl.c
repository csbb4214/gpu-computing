#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clu_errcheck.h"
#include "clu_setup.h"

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

int main(void) {
	clu_env env;
	if(clu_initialize(&env) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		return EXIT_FAILURE;
	}

#ifndef FLOAT
	if(!clu_check_double_support(env.device_id)) {
		fprintf(stderr, "Error: Device does not support double precision (cl_khr_fp64)\n");
		clu_release(&env);
		return EXIT_FAILURE;
	}
#endif


	const char kernel_path[] = "./jacobi.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, NULL);
	if(program == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);

	// init matrices
	memset(u, 0, sizeof(u));
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			f[i][j] = init_func(i, j);
		}
	}

	const VALUE factor = pow((VALUE)1 / N, 2);
	const size_t bytes = sizeof(VALUE) * N * N;

	cl_mem buf_u = clCreateBuffer(env.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_tmp = clCreateBuffer(env.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_f = clCreateBuffer(env.context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_f, CL_TRUE, 0, bytes, f, 0, NULL, NULL));


	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_f));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(VALUE), (void*)&factor));

	const double start_time = omp_get_wtime();
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_tmp, CL_TRUE, 0, bytes, u, 0, NULL, NULL));

	const size_t global_work_size[2] = {(size_t)N, (size_t)N};
	const size_t local_work_size[2] = {2, 128};
#if VERSION == 1
	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));

		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_tmp));

		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));

		CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_tmp, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
	}
#else
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_tmp));

		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
		cl_mem temp = buf_u;
		buf_u = buf_tmp;
		buf_tmp = temp;
	}
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
#endif

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

	// Calculate checksum
	VALUE checksum = 0;
	for(int i = 1; i < N - 1; i++) {
		for(int j = 1; j < N - 1; j++) {
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

	CLU_ERRCHECK(clFlush(env.command_queue));
	CLU_ERRCHECK(clFinish(env.command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(buf_u));
	CLU_ERRCHECK(clReleaseMemObject(buf_tmp));
	CLU_ERRCHECK(clReleaseMemObject(buf_f));

	free(source_str);

	clu_release(&env);

	return EXIT_SUCCESS;
}
