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

VALUE u[N][N], tmp[N][N], f[N][N];

VALUE init_func(int x, int y) {
	return 40 * sin((VALUE)(16 * (2 * x - 1) * y));
}

int main(void) {
	/* setup and init */
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	if(clu_initialize(&env, queue_properties) != 0) {
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

	// Timing variables
	cl_ulong write_time[3];
	cl_ulong total_write_time = 0;
	cl_ulong total_kernel_time = 0;
	cl_ulong total_read_time = 0;

	// Queue timing variable
	cl_ulong total_queue_time = 0;

	cl_mem buf_u = clCreateBuffer(env.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_tmp = clCreateBuffer(env.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_f = clCreateBuffer(env.context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);

	cl_event event_time_write[3];
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_f, CL_TRUE, 0, bytes, f, 0, NULL, &event_time_write[0]));
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_tmp, CL_TRUE, 0, bytes, u, 0, NULL, &event_time_write[1]));
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, &event_time_write[2]));

	CLU_ERRCHECK(clWaitForEvents(3, event_time_write));

	// Calculate write times and queue times for write operations
	for(int i = 0; i < 3; i++) {
		cl_ulong queuedtime, starttime, endtime;
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time_write[i], CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedtime, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time_write[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time_write[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL));

		cl_ulong elapsed = (cl_ulong)(endtime - starttime);
		cl_ulong queue_elapsed = (cl_ulong)(starttime - queuedtime);

		write_time[i] = elapsed;
		total_write_time += elapsed;
		total_queue_time += queue_elapsed;
	}

	for(int i = 0; i < 3; i++) {
		CLU_ERRCHECK(clReleaseEvent(event_time_write[i]));
	}

	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buf_f));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(VALUE), (void*)&factor));

	const size_t global_work_size[2] = {(size_t)N, (size_t)N};
	const size_t local_work_size[2] = {2, 128};

	char detail_filename[256];
#ifdef FLOAT
	snprintf(detail_filename, sizeof(detail_filename), "kernel_times_N%d_IT%d_float.csv", N, IT);
	const char* prec = "float";
#else
	snprintf(detail_filename, sizeof(detail_filename), "kernel_times_N%d_IT%d_double.csv", N, IT);
	const char* prec = "double";
#endif

	FILE* detail_file = fopen(detail_filename, "w");
	if(detail_file != NULL) { fprintf(detail_file, "iteration,kernel_time_ms,queue_time_ms\n"); }

	for(int it = 0; it < IT; it++) {
		CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buf_u));
		CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buf_tmp));

		cl_event event_time;
		CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &event_time));

		CLU_ERRCHECK(clWaitForEvents(1, &event_time));

		cl_ulong queuedtime, starttime, endtime;
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedtime, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime, NULL));
		CLU_ERRCHECK(clGetEventProfilingInfo(event_time, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime, NULL));

		cl_ulong queue_elapsed = starttime - queuedtime;
		cl_ulong elapsed = endtime - starttime;

		total_queue_time += queue_elapsed;
		total_kernel_time += elapsed;

		if(detail_file != NULL) { fprintf(detail_file, "%d,%.6f,%.6f\n", it, (double)(elapsed * 1e-6), (double)(queue_elapsed * 1e-6)); }

		CLU_ERRCHECK(clReleaseEvent(event_time));

		cl_mem temp = buf_u;
		buf_u = buf_tmp;
		buf_tmp = temp;
	}
	if(detail_file != NULL) { fclose(detail_file); }

	cl_event event_time_read;
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_u, CL_TRUE, 0, bytes, u, 0, NULL, &event_time_read));

	// Queue time for read operation
	cl_ulong queuedtime_read, starttime_read, endtime_read;
	CLU_ERRCHECK(clGetEventProfilingInfo(event_time_read, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queuedtime_read, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(event_time_read, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &starttime_read, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(event_time_read, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endtime_read, NULL));

	cl_ulong read_queue_elapsed = starttime_read - queuedtime_read;
	total_queue_time += read_queue_elapsed;
	total_read_time = (cl_ulong)(endtime_read - starttime_read);

	CLU_ERRCHECK(clReleaseEvent(event_time_read));

	// Calculate checksum
	VALUE checksum = 0;
	for(int i = 1; i < N - 1; i++) {
		for(int j = 1; j < N - 1; j++) {
			checksum += u[i][j];
		}
	}

	// Average queue time
	int total_operations = 3 + IT + 1; // 3 writes + IT kernels + 1 read
	double average_queue_time = (double)(total_queue_time * 1e-6) / total_operations;

	printf("%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", prec, N, IT, (double)(total_kernel_time * 1e-6), (double)(total_read_time * 1e-6),
	    (double)(total_write_time * 1e-6), (double)(write_time[0] * 1e-6), (double)(write_time[1] * 1e-6), (double)(write_time[2] * 1e-6), average_queue_time);

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