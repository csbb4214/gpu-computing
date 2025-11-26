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
#define VERSION 1
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
#define ONE (1.0f)
#define PRINT_FMT "%.3f"
#else
#define VALUE int
#define ZERO (0)
#define ONE (1)
#define PRINT_FMT "%d"
#endif

VALUE result = ZERO;

int main(void) {
	// ========== Initialize host matrices ==========
	VALUE* arr = (VALUE*)malloc(sizeof(VALUE) * N);
	if(!arr) {
		printf("malloc for arr failed!");
		return EXIT_FAILURE;
	}
	VALUE* partial_results = (VALUE*)calloc(N, sizeof(VALUE));
	if(!partial_results) {
		printf("calloc for partial_results failed!");
		return EXIT_FAILURE;
	}

	for(size_t i = 0; i < N; i++) {
		arr[i] = ONE;
	}

#if VERSION == 1
	const double start_time = omp_get_wtime();

	for(size_t i = 0; i < N; i++) {
		result += arr[i];
	}

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;
#else
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

#ifdef FLOAT
	const char* options = "-DFLOAT=1";
#else
	const char* options = NULL;
#endif
	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, options);
	if(program == NULL) {
		free(source_str);
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;
	cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);

	// ========== Setup kernel parameters ==========
	const size_t bytes = sizeof(VALUE) * N;
	const size_t global_work_size[1] = {(size_t)N};
	const size_t local_work_size[1] = {256};

	// ========== Create device buffers ==========
	cl_mem buf_arr = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_partial_results = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	// ========== Write data to device ==========
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_arr, CL_TRUE, 0, bytes, arr, 0, NULL, NULL));
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_partial_results, CL_TRUE, 0, bytes, partial_results, 0, NULL, NULL));

	// ========== Timing Setup ==========

#if VERSION == 2
	cl_event kernel_event;
#else
	int stages = (int)ceil(log2((double)global_work_size[0] / (double)local_work_size[0]));
	cl_event* kernel_events = (cl_event*)malloc(stages * sizeof(cl_event));
	double* kernel_times = (double*)malloc(stages * sizeof(double));

	if(kernel_events == NULL || kernel_times == NULL) {
		fprintf(stderr, "Failed to allocate memory for timing arrays\n");
		free(kernel_events);
		free(kernel_times);
		CLU_ERRCHECK(clFlush(env.command_queue));
		CLU_ERRCHECK(clFinish(env.command_queue));
		CLU_ERRCHECK(clReleaseKernel(kernel));
		CLU_ERRCHECK(clReleaseProgram(program));
		CLU_ERRCHECK(clReleaseMemObject(buf_arr));
		free(source_str);
		clu_release(&env);

		return EXIT_FAILURE;
	}
#endif

	// ========== Enqueue kernels ==========
	const size_t local_mem_size = local_work_size[0] * sizeof(VALUE);

#if VERSION == 2
	const cl_int length = N;

	CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_arr));
	CLU_ERRCHECK(clSetKernelArg(kernel, 1, local_mem_size, NULL));
	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &length));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buf_partial_results));

	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 1, NULL, global_work_size, local_work_size, 0, NULL, &kernel_event));
#else
	int stage = 0;
	size_t stage_input_size = N;

	cl_mem buf_in = buf_arr;
	cl_mem buf_out = buf_partial_results;

	while(stage_input_size > 1) {
		size_t num_groups = (stage_input_size + local_work_size[0] - 1) / local_work_size[0];
		size_t stage_global = num_groups * local_work_size[0];

		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_in));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, local_mem_size, NULL));
		CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_int), &stage_input_size));
		CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buf_out));

		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 1, NULL, &stage_global, local_work_size, 0, NULL, &kernel_events[stage]));

		cl_mem tmp = buf_in;
		buf_in = buf_out;
		buf_out = tmp;

		stage_input_size = num_groups;
		stage++;
	}
#endif

	// ========== Extract kernel timing information ==========
	cl_ulong kernel_time = 0;
	cl_ulong start, end;
#if VERSION == 2
	CLU_ERRCHECK(clWaitForEvents(1, &kernel_event));
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));
	kernel_time = end - start;
	const double elapsed_ms = (double)(kernel_time * 1e-6);
	CLU_ERRCHECK(clReleaseEvent(kernel_event));
#else
	CLU_ERRCHECK(clWaitForEvents(1, &kernel_events[stage - 1]));
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events[stage - 1], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));
	kernel_time = end - start;
	const double elapsed_ms = (double)(kernel_time * 1e-6);
	for(int i = 0; i < stage; i++) {
		CLU_ERRCHECK(clReleaseEvent(kernel_events[i]));
	}
#endif

	// ========== Read result back to host ==========

#if VERSION == 2
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_partial_results, CL_TRUE, 0, bytes, partial_results, 0, NULL, NULL));

	for(size_t i = 0; i < N; i++) {
		result += partial_results[i];
	}
#else
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_in, CL_TRUE, 0, sizeof(VALUE), &result, 0, NULL, NULL));
#endif

#endif

	// ========== Print results ==========

#ifdef FLOAT
	const char* prec = "float";
#else
	const char* prec = "int";
#endif

#if VERSION == 1
	printf("sequential_reduction,%s,%d," PRINT_FMT ",%.3f\n", prec, N, result, elapsed_ms);
#elif VERSION == 2
	printf("parallel_reduction,%s,%d," PRINT_FMT ",%.3f\n", prec, N, result, elapsed_ms);
#elif VERSION == 3
	printf("multistage_reduction,%s,%d," PRINT_FMT ",%.3f\n", prec, N, result, elapsed_ms);
#endif

#if VERSION != 1
	// ========== Cleanup ==========
	CLU_ERRCHECK(clFlush(env.command_queue));
	CLU_ERRCHECK(clFinish(env.command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(buf_arr));
	CLU_ERRCHECK(clReleaseMemObject(buf_partial_results));
	free(source_str);
	clu_release(&env);
#endif

	free(arr);
	free(partial_results);

	return EXIT_SUCCESS;
}