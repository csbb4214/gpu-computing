#include "clu_errcheck.h"
#include "clu_setup.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1000
#endif

#define M N
#define K N

#if defined(USE_DOUBLE)
#define VALUE double
#define KERNEL_NAME "matrix_mul_double_2cols"
#else
#define VALUE float
#define KERNEL_NAME "matrix_mul_float_2cols"
#endif

// how many columns each work-item computes
#define COLS_PER_THREAD 2

// local work-group size (tuned for GPUs)
#define TILE_X 8
#define TILE_Y 32

// host matrices
VALUE A[N * M];
VALUE B[M * K];
VALUE C[N * K];

int main(void) {
	// ====== Initialize host matrices ======
	for(size_t i = 0; i < (size_t)N; i++) {
		for(size_t j = 0; j < (size_t)M; j++) {
			A[i * M + j] = (VALUE)1.0;
		}
	}

	for(size_t i = 0; i < (size_t)M; i++) {
		for(size_t j = 0; j < (size_t)K; j++) {
			B[i * K + j] = (VALUE)1.0;
		}
	}

	for(size_t i = 0; i < (size_t)N; i++) {
		for(size_t j = 0; j < (size_t)K; j++) {
			C[i * K + j] = (VALUE)0.0;
		}
	}

	// ====== Initialization ======
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

	if(clu_initialize(&env, queue_properties) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		return EXIT_FAILURE;
	}

	// print device name (quick check CPU vs GPU)
	{
		char device_name[256] = {0};
		clGetDeviceInfo(env.device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
		printf("Using OpenCL device: %s\n", device_name);
	}

#if defined(USE_DOUBLE)
	if(!clu_check_double_support(env.device_id)) {
		fprintf(stderr, "Device does not support double precision.\n");
		clu_release(&env);
		return EXIT_FAILURE;
	}
#endif

	// ====== Load and compile kernel ======
	const char kernel_path[] = "./matrix_mul_test.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

#if defined(USE_DOUBLE)
	// enable double and math optimizations
	const char* build_options = "-DUSE_DOUBLE=1 -cl-mad-enable -cl-fast-relaxed-math";
#else
	// enable math optimizations
	const char* build_options = "-cl-mad-enable -cl-fast-relaxed-math";
#endif

	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, build_options);
	free(source_str);

	if(program == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);

	// ====== Create buffers ======
	cl_mem bufA = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * (size_t)N * (size_t)M, A, &err);
	CLU_ERRCHECK(err);

	cl_mem bufB = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * (size_t)M * (size_t)K, B, &err);
	CLU_ERRCHECK(err);

	cl_mem bufC = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, sizeof(VALUE) * (size_t)N * (size_t)K, NULL, &err);
	CLU_ERRCHECK(err);

	// ====== Set kernel arguments ======
	const cl_int N_arg = (cl_int)N;
	const cl_int M_arg = (cl_int)M;
	const cl_int K_arg = (cl_int)K;

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(kernel, 3, sizeof(cl_int), &N_arg);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(kernel, 4, sizeof(cl_int), &M_arg);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(kernel, 5, sizeof(cl_int), &K_arg);
	CLU_ERRCHECK(err);

	// ====== Launch kernel ======
	// use 2 columns per thread, round up global size to multiples of TILE
	const size_t num_col_groups = ((size_t)K + (size_t)COLS_PER_THREAD - 1) / (size_t)COLS_PER_THREAD;

	size_t global_work_size[2];
	global_work_size[0] = ((size_t)N + (size_t)TILE_X - 1) / (size_t)TILE_X * (size_t)TILE_X;
	global_work_size[1] = (num_col_groups + (size_t)TILE_Y - 1) / (size_t)TILE_Y * (size_t)TILE_Y;

	const size_t local_work_size[2] = {TILE_X, TILE_Y}; // explicit local size

	const double start_time = omp_get_wtime();

	err = clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	CLU_ERRCHECK(err);

	err = clFinish(env.command_queue);
	CLU_ERRCHECK(err);

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

	// ====== Read back result ======
	err = clEnqueueReadBuffer(env.command_queue, bufC, CL_TRUE, 0, sizeof(VALUE) * (size_t)N * (size_t)K, C, 0, NULL, NULL);
	CLU_ERRCHECK(err);

	printf("C[0,0] = %f, time = %.3f ms\n", (double)C[0], elapsed_ms);

	// ====== Cleanup ======
	clReleaseMemObject(bufC);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufA);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clu_release(&env);

	return EXIT_SUCCESS;
}
