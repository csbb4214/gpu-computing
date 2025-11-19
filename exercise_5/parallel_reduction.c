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

#ifndef VERSION
#define VERSION 2
#endif

#if VERSION == 2
#define KERNEL_NAME "parallel_reduction"
#endif
#if VERSION == 3
#define KERNEL_NAME "multistage_reduction"
#endif

#ifdef FLOAT
#define VALUE float
#define ZERO (0.0f)
#else
#define VALUE int
#define ZERO (0)
#endif

VALUE u[N];
VALUE partial_results[N];
VALUE result = ZERO;

int main(void) {
	// ========== Initialization ==========
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	if(clu_initialize(&env, queue_properties) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		return EXIT_FAILURE;
	}

	// ========== Load and compile kernel ==========
	const char kernel_path[] = "./reduction.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, NULL);
	if(program == NULL) {
		free(source_str);
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);

	// ========== Initialize host matrices ==========
	memset(u, 1, sizeof(u));
	memset(partial_results, 0, sizeof(partial_results));

	// ========== Setup kernel parameters ==========
	const size_t bytes = sizeof(VALUE) * N;
	const size_t global_work_size[1] = {(size_t)N};
	const size_t local_work_size[1] = {256};

	// ========== Create device buffers ==========
	cl_mem buf_u = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_partial_results = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	// ========== Write data to device ==========
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_partial_results, CL_TRUE, 0, bytes, u, 0, NULL, NULL));

	// ========== Set kernel arguments ==========
	const size_t local_mem_size = local_work_size[0] * sizeof(VALUE);
	const cl_int length = N;

	CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
	CLU_ERRCHECK(clSetKernelArg(kernel, 1, local_mem_size, NULL));
	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &length));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buf_partial_results));

	// ========== Allocate timing arrays ==========
	cl_event* kernel_events = (cl_event*)malloc(sizeof(cl_event));
	double* kernel_times = (double*)malloc(sizeof(double));

	if(kernel_events == NULL || kernel_times == NULL) {
		fprintf(stderr, "Failed to allocate memory for timing arrays\n");
		free(kernel_events);
		free(kernel_times);
		CLU_ERRCHECK(clFlush(env.command_queue));
		CLU_ERRCHECK(clFinish(env.command_queue));
		CLU_ERRCHECK(clReleaseKernel(kernel));
		CLU_ERRCHECK(clReleaseProgram(program));
		CLU_ERRCHECK(clReleaseMemObject(buf_u));
		free(source_str);
		clu_release(&env);

		return EXIT_FAILURE;
	}

	// ========== Enqueue kernels ==========

#if VERSION == 2
	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_events));
	CLU_ERRCHECK(clWaitForEvents(1, kernel_events));
#endif

	// ========== Extract kernel timing information ==========
	cl_ulong elapsed_ms = 0;
	cl_ulong start, end;
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));
	elapsed_ms = end - start;
	CLU_ERRCHECK(clReleaseEvent(kernel_events));

	// ========== Read result back to host ==========

#if VERSION == 2
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_partial_results, CL_TRUE, 0, sizeof(VALUE), &partial_results, 0, NULL, NULL));
#endif

	// ========== Check Results ==========

#if VERSION == 2
	for(size_t i = 0; i < N; i++) {
		result += partial_results[i];
	}
#endif

	// ========== Print summary statistics ==========

#ifdef FLOAT
	const char* prec = "float";
#else
	const char* prec = "int";
#endif

#if VERSION == 1
	printf("sequential_reduction,%s,%d,%s,%.3f\n", prec, N, abs(result - N), elapsed_ms);
#elif VERSION == 2
	printf("parallel_reduction,%s,%d,%s,%.3f\n", prec, N, abs(result - N), elapsed_ms);
#elif VERSION == 3
	printf("multistage_reduction,%s,%d,%s,%.3f\n", prec, N, abs(result - N), elapsed_ms);
#endif

	// ========== Cleanup ==========
	CLU_ERRCHECK(clFlush(env.command_queue));
	CLU_ERRCHECK(clFinish(env.command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(buf_u));
	CLU_ERRCHECK(clReleaseMemObject(buf_partial_results));
	free(source_str);
	clu_release(&env);

	return EXIT_SUCCESS;
}