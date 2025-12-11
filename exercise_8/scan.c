#include <assert.h>
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "clu_errcheck.h"
#include "clu_setup.h"

#ifndef N
#define N 1024
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

// ========== Sequential Inclusive Scan ==========
void inclusiveScan(const VALUE* input, VALUE* output, size_t size) {
	if(size == 0) return;

	output[0] = input[0];
	for(size_t i = 1; i < size; i++) {
		output[i] = output[i - 1] + input[i];
	}
}

// ========== Comparison for validation ==========
int compareArrays(const VALUE* arr1, const VALUE* arr2, size_t size) {
	for(size_t i = 0; i < size; i++) {
#ifdef FLOAT
		if(fabs(arr1[i] - arr2[i]) > 0.001f) {
			printf("Mismatch at index %zu: %.3f != %.3f\n", i, arr1[i], arr2[i]);
			return 0;
		}
#else
		if(arr1[i] != arr2[i]) {
			printf("Mismatch at index %zu: %d != %d\n", i, arr1[i], arr2[i]);
			return 0;
		}
#endif
	}
	return 1;
}

int main(void) {
	// ========== Initialize arrays ==========
	VALUE* input = (VALUE*)malloc(sizeof(VALUE) * N);
	VALUE* output_sequential = (VALUE*)malloc(sizeof(VALUE) * N);
	if(!input || !output_sequential) {
		fprintf(stderr, "malloc failed!\n");
		return EXIT_FAILURE;
	}

	// Initialize with random values
	srand(time(NULL));
	for(size_t i = 0; i < N; i++) {
#ifdef FLOAT
		input[i] = (VALUE)(rand() % 100) / 10.0f;
#else
		input[i] = (VALUE)(rand() % 10);
#endif
	}

	// ========== Sequential Version -> Compute result for validation ==========
	printf("\n--- Sequential Inclusive Scan ---\n");
	const double start_time = omp_get_wtime();
	inclusiveScan(input, output_sequential, N);
	double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

	printf("Sequential Time: %.3f ms\n", elapsed_ms);

	// ========== OpenCL Version ==========

#ifndef OPT
	printf("\n--- OpenCL Inclusive Scan (Hillis & Steele) ---\n");
#else
	printf("\n--- OpenCL Inclusive Scan (Optimized) ---\n");
#endif

	// Initialize OpenCL
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	if(clu_initialize(&env, queue_properties) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		free(input);
		free(output_sequential);
		return EXIT_FAILURE;
	}

	// Load and compile kernel
	const char kernel_path[] = "./scan.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		free(input);
		free(output_sequential);
		return EXIT_FAILURE;
	}

#ifdef FLOAT
#ifdef OPT
	const char* options = "-DFLOAT=1 -DOPT=1";
#else
	const char* options = "-DFLOAT=1";
#endif
#else
#ifdef OPT
	const char* options = "-DOPT=1";
#else
	const char* options = NULL;
#endif
#endif
	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, options);
	if(program == NULL) {
		free(source_str);
		clu_release(&env);
		free(input);
		free(output_sequential);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;
#ifdef OPT
	cl_kernel kernel_scan = clCreateKernel(program, "improved_scan", &err);
	CLU_ERRCHECK(err);
#else
	cl_kernel kernel_scan = clCreateKernel(program, "hillis_steele_scan", &err);
	CLU_ERRCHECK(err);
#endif
	cl_kernel kernel_add = clCreateKernel(program, "add_block_sums", &err);
	CLU_ERRCHECK(err);

	// Setup kernel parameters
	const size_t bytes = sizeof(VALUE) * N;
	const size_t local_work_size = 256;
#ifdef OPT
	const size_t elements_per_thread = 2;
	const size_t elements_per_block = local_work_size * elements_per_thread;
	const size_t num_blocks = (N + elements_per_block - 1) / elements_per_block;
	const size_t global_work_size = num_blocks * local_work_size;
#else
	const size_t global_work_size = ((N + local_work_size - 1) / local_work_size) * local_work_size;
	const size_t num_blocks = global_work_size / local_work_size;
#endif

	VALUE* output_opencl = (VALUE*)malloc(sizeof(VALUE) * N);
	VALUE* block_sums_host = (VALUE*)malloc(sizeof(VALUE) * num_blocks);
	VALUE* block_sums_scanned = (VALUE*)malloc(sizeof(VALUE) * num_blocks);
	if(!output_opencl || !block_sums_host || !block_sums_scanned) {
		fprintf(stderr, "malloc failed!\n");
		CLU_ERRCHECK(clReleaseKernel(kernel_scan));
		CLU_ERRCHECK(clReleaseKernel(kernel_add));
		CLU_ERRCHECK(clReleaseProgram(program));
		free(source_str);
		clu_release(&env);
		free(input);
		free(output_sequential);
		free(output_opencl);
		free(block_sums_host);
		free(block_sums_scanned);
		return EXIT_FAILURE;
	}

	// Create device buffers
	cl_mem buf_input = clCreateBuffer(env.context, CL_MEM_READ_ONLY, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_output = clCreateBuffer(env.context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_block_sums = clCreateBuffer(env.context, CL_MEM_READ_WRITE, sizeof(VALUE) * num_blocks, NULL, &err);
	CLU_ERRCHECK(err);
	cl_mem buf_block_sums_scanned = clCreateBuffer(env.context, CL_MEM_READ_ONLY, sizeof(VALUE) * num_blocks, NULL, &err);
	CLU_ERRCHECK(err);

	// Write data to device
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_input, CL_TRUE, 0, bytes, input, 0, NULL, NULL));

	// Phase 1: Local scan within each work-group
	const cl_int n = N;
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 0, sizeof(cl_mem), &buf_output));
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 1, sizeof(cl_mem), &buf_input));
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 2, sizeof(cl_int), &n));
#ifdef OPT
	const size_t local_mem_size = (local_work_size * 2 + ((local_work_size * 2) >> 5)) * sizeof(VALUE);
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 3, local_mem_size, NULL));
#else
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 3, 2 * local_work_size * sizeof(VALUE), NULL));
#endif
	CLU_ERRCHECK(clSetKernelArg(kernel_scan, 4, sizeof(cl_mem), &buf_block_sums));

	cl_event event_phase1;
	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel_scan, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event_phase1));

	// Read block sums
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_block_sums, CL_TRUE, 0, sizeof(VALUE) * num_blocks, block_sums_host, 0, NULL, NULL));

	// Phase 2: Scan the block sums on CPU
	inclusiveScan(block_sums_host, block_sums_scanned, num_blocks);

	// Write scanned block sums back
	CLU_ERRCHECK(clEnqueueWriteBuffer(env.command_queue, buf_block_sums_scanned, CL_TRUE, 0, sizeof(VALUE) * num_blocks, block_sums_scanned, 0, NULL, NULL));

	// Phase 3: Add block sums to all elements
	CLU_ERRCHECK(clSetKernelArg(kernel_add, 0, sizeof(cl_mem), &buf_output));
	CLU_ERRCHECK(clSetKernelArg(kernel_add, 1, sizeof(cl_mem), &buf_block_sums_scanned));
	CLU_ERRCHECK(clSetKernelArg(kernel_add, 2, sizeof(cl_int), &n));

	cl_event event_phase3;
	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel_add, 1, NULL, &global_work_size, &local_work_size, 0, NULL, &event_phase3));

	// Get timing information (total time from phase 1 to phase 3)
	CLU_ERRCHECK(clWaitForEvents(1, &event_phase3));
	cl_ulong start, end;
	CLU_ERRCHECK(clGetEventProfilingInfo(event_phase1, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(event_phase3, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL));
	elapsed_ms = (double)((end - start) * 1e-6);

	// Read result back
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_output, CL_TRUE, 0, bytes, output_opencl, 0, NULL, NULL));

	printf("OpenCL Time: %.3f ms\n", elapsed_ms);

	// Validate result
	printf("\n--- Validation ---\n");
	if(compareArrays(output_opencl, output_sequential, N)) {
		printf("PASSED: OpenCL result matches sequential result\n");
	} else {
		printf("FAILED: Results do not match\n");
	}

	// Cleanup
	CLU_ERRCHECK(clReleaseEvent(event_phase1));
	CLU_ERRCHECK(clReleaseEvent(event_phase3));
	CLU_ERRCHECK(clFlush(env.command_queue));
	CLU_ERRCHECK(clFinish(env.command_queue));
	CLU_ERRCHECK(clReleaseKernel(kernel_scan));
	CLU_ERRCHECK(clReleaseKernel(kernel_add));
	CLU_ERRCHECK(clReleaseProgram(program));
	CLU_ERRCHECK(clReleaseMemObject(buf_input));
	CLU_ERRCHECK(clReleaseMemObject(buf_output));
	CLU_ERRCHECK(clReleaseMemObject(buf_block_sums));
	CLU_ERRCHECK(clReleaseMemObject(buf_block_sums_scanned));
	free(source_str);
	clu_release(&env);
	free(output_opencl);
	free(block_sums_host);
	free(block_sums_scanned);

	free(input);
	free(output_sequential);

	return EXIT_SUCCESS;
}