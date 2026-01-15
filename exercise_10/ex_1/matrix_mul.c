#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "clu_errcheck.h"
#include "clu_setup.h"

#ifndef N
#define N 1024
#endif

#if defined(USE_DOUBLE)
#define VALUE double
#define PRECISION_STR "double"
#else
#define VALUE float
#define PRECISION_STR "float"
#endif

const cl_int n = N;

// ---------------- CPU Reference (optional) ----------------
void cpu_matrix_mul(const VALUE* A, const VALUE* B, VALUE* C, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			VALUE sum = 0;
			for(int k = 0; k < n; k++) {
				sum += A[i * n + k] * B[k * n + j];
			}
			C[i * n + j] = sum;
		}
	}
}

// ---------------- Load kernel ----------------
static char* load_kernel(const char* path, size_t* out_size) {
	char* src = clu_load_kernel_source(path, out_size);
	if(!src) {
		fprintf(stderr, "Failed to load kernel source '%s'\n", path);
		return NULL;
	}
	return src;
}

int main(void) {
	// ---------------- Allocate matrices ----------------
	VALUE* A = malloc(sizeof(VALUE) * N * N);
	VALUE* B = malloc(sizeof(VALUE) * N * N);
	VALUE* C = malloc(sizeof(VALUE) * N * N);
	VALUE* C_ref = malloc(sizeof(VALUE) * N * N);

	if(!A || !B || !C || !C_ref) {
		fprintf(stderr, "Out of memory!\n");
		free(A);
		free(B);
		free(C);
		free(C_ref);
		return EXIT_FAILURE;
	}

	// Initialize matrices
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			A[i * N + j] = (VALUE)(i + 1);
			B[i * N + j] = (i == j) ? (VALUE)1 : (VALUE)0;
			C[i * N + j] = 0;
		}
	}

	// Optional: CPU reference for validation
	cpu_matrix_mul(A, B, C_ref, N);

	// ---------------- OpenCL Setup ----------------
	clu_env env = {0};
	cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	if(clu_initialize(&env, props) != 0) {
		fprintf(stderr, "OpenCL initialization failed\n");
		free(A);
		free(B);
		free(C);
		free(C_ref);
		return EXIT_FAILURE;
	}

	size_t kernel_size = 0;
	char* kernel_src = load_kernel("./matrix_mul.cl", &kernel_size);
	if(!kernel_src) {
		clu_release(&env);
		free(A);
		free(B);
		free(C);
		free(C_ref);
		return EXIT_FAILURE;
	}

	const char* build_opts =
#if defined(USE_DOUBLE)
	    "-DUSE_DOUBLE";
#else
	    "-DFLOAT";
#endif

	cl_program program = clu_create_program(env.context, env.device_id, kernel_src, kernel_size, build_opts);
	if(!program) {
		free(kernel_src);
		clu_release(&env);
		free(A);
		free(B);
		free(C);
		free(C_ref);
		return EXIT_FAILURE;
	}

	cl_int err;
	cl_kernel kernel = clCreateKernel(program,
#if defined(USE_DOUBLE)
	    "matrix_mul_double",
#else
	    "matrix_mul_float",
#endif
	    &err);
	CLU_ERRCHECK(err);

	// ---------------- Buffers ----------------
	cl_mem bufA = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * N * N, A, &err);
	CLU_ERRCHECK(err);
	cl_mem bufB = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * N * N, B, &err);
	CLU_ERRCHECK(err);
	cl_mem bufC = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, sizeof(VALUE) * N * N, NULL, &err);
	CLU_ERRCHECK(err);

	// ---------------- Kernel Args ----------------
	CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA));
	CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB));
	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC));
	CLU_ERRCHECK(clSetKernelArg(kernel, 3, sizeof(cl_int), &n));
	CLU_ERRCHECK(clSetKernelArg(kernel, 4, sizeof(cl_int), &n));

	size_t global[2] = {(size_t)N, (size_t)N};

	// ---------------- Kernel enqueue ----------------
	cl_event kernel_event;
	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global, NULL, 0, NULL, &kernel_event));

	// ---------------- Read back (Output) ----------------
	cl_event read_event;
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, bufC, CL_FALSE, 0, sizeof(VALUE) * N * N, C, 1, &kernel_event, &read_event));

	CLU_ERRCHECK(clWaitForEvents(1, &read_event));

	// ---------------- Profiling ----------------
	cl_ulong t_start, t_end;
	CLU_ERRCHECK(clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(t_start), &t_start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(t_end), &t_end, NULL));

	double elapsed_ms = (t_end - t_start) * 1e-6;

	// ---------------- Read back ----------------
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, bufC, CL_TRUE, 0, sizeof(VALUE) * N * N, C, 0, NULL, NULL));

	// ---------------- Validation ----------------
	int correct = 1;
	for(int i = 0; i < N * N; i++) {
		if(C[i] != C_ref[i]) {
			correct = 0;
			break;
		}
	}

	// ---------------- Output CSV ----------------
	printf("%s,%d,%.3f\n", PRECISION_STR, N, elapsed_ms);

	// ---------------- Cleanup ----------------
	CLU_ERRCHECK(clReleaseMemObject(bufA));
	CLU_ERRCHECK(clReleaseMemObject(bufB));
	CLU_ERRCHECK(clReleaseMemObject(bufC));
	CLU_ERRCHECK(clReleaseKernel(kernel));
	CLU_ERRCHECK(clReleaseProgram(program));
	clu_release(&env);

	free(kernel_src);
	free(A);
	free(B);
	free(C);
	free(C_ref);

	return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
