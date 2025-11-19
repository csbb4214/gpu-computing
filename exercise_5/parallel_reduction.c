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

	// ========== Setup kernel parameters ==========
	const size_t bytes = sizeof(VALUE) * N;
	const size_t global_work_size[1] = {(size_t)N};
	const size_t local_work_size[1] = {256};

	// ========== Create device buffers ==========
	cl_mem buf_u = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	// ========== Write data to device ==========
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));

	// ========== Set constant kernel arguments ==========

	const cl_int pitch = (cl_int)local_work_size[0] + 2;
	const cl_int dim = N;
	const size_t local_mem_size = pitch * (local_work_size[1] + 2) * sizeof(VALUE);

	CLU_ERRCHECK(clSetKernelArg(kernel, 3, local_mem_size, NULL));
	CLU_ERRCHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &pitch));
	CLU_ERRCHECK(clSetKernelArg(kernel, 5, sizeof(cl_int), &dim));
	CLU_ERRCHECK(clSetKernelArg(kernel, 6, sizeof(VALUE), (void*)&factor));


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
	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_result));

#ifdef DETAILED_TIMING
		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &kernel_events[it]));
#else
		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL));
#endif

		cl_mem temp = buf_u;
		buf_u = buf_tmp;
		buf_tmp = temp;
	}

#ifdef DETAILED_TIMING
	CLU_ERRCHECK(clWaitForEvents(IT, kernel_events));
#endif

#ifdef DETAILED_TIMING
	// ========== Extract kernel timing information ==========
	cl_ulong total_kernel_time = 0;

	for(int it = 0; it < IT; it++) {
		cl_ulong queued, start, end;
		CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events[it], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events[it], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(kernel_events[it], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));

		cl_ulong queue_elapsed = start - queued;
		cl_ulong elapsed = end - start;

		total_queue_time += queue_elapsed;
		total_kernel_time += elapsed;

		kernel_times[it] = (double)(elapsed * 1e-6);
		queue_times[it] = (double)(queue_elapsed * 1e-6);

		CLU_ERRCHECK(clReleaseEvent(kernel_events[it]));
	}
#endif

#ifdef FLOAT
	const char* prec = "float";
#else
	const char* prec = "int";
#endif


#ifdef DETAILED_TIMING
	// ========== Write timing data to CSV file ==========
	char detail_filename[256];
#ifdef FLOAT
	snprintf(detail_filename, sizeof(detail_filename), "kernel_times_N%d_IT%d_float.csv", N, IT);
#else
	snprintf(detail_filename, sizeof(detail_filename), "kernel_times_N%d_IT%d_double.csv", N, IT);
#endif

	FILE* detail_file = fopen(detail_filename, "w");
	if(detail_file != NULL) {
		fprintf(detail_file, "iteration,kernel_time_ms,queue_time_ms\n");
		for(int it = 0; it < IT; it++) {
			fprintf(detail_file, "%d,%.6f,%.6f\n", it, kernel_times[it], queue_times[it]);
		}
		fclose(detail_file);
	}

	free(kernel_events);
	free(kernel_times);
	free(queue_times);
#endif

	// ========== Read result back to host ==========

#ifdef DETAILED_TIMING
	cl_event read_event;
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, &read_event));
#else
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, NULL));
#endif
	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;


#ifdef DETAILED_TIMING
	// ========== Measure read times ==========
	cl_ulong read_queued, read_start, read_end;
	CLU_ERRCHECK(clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &read_queued, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &read_start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &read_end, NULL));

	total_queue_time += (read_start - read_queued);
	cl_ulong total_read_time = read_end - read_start;

	CLU_ERRCHECK(clReleaseEvent(read_event));
#endif

	// ========== Check Results ==========


	// ========== Print summary statistics ==========

	int total_operations = 3 + IT + 1; // 3 writes + IT kernels + 1 read
	double avg_queue_time = (double)(total_queue_time * 1e-6) / total_operations;

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
	CLU_ERRCHECK(clReleaseMemObject(buf_f));
	free(source_str);
	clu_release(&env);

	return EXIT_SUCCESS;
}