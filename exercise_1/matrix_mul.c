#include <CL/cl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef N
#define N 1000
#endif

#define M N
#define K N

#define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X, Y) ((X) > (Y) ? (X) : (Y))

#define VALUE double

VALUE A[N * M];
VALUE B[M * K];
VALUE C[N * K];

const cl_int m = M;
const cl_int k = K;

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

	size_t read = fread(buffer, 1U, (size_t)length, file);
	if(read != (size_t)length) {
		fprintf(stderr, "Failed to read kernel source '%s'.\n", path);
		free(buffer);
		fclose(file);
		return NULL;
	}

	buffer[length] = '\0';
	if(out_size) *out_size = read;

	fclose(file);
	return buffer;
}

int main() {
	// initialize C matrix to zero
	memset(C, 0, sizeof(C));

	// A contains real values
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < M; j++) {
			A[i * N + j] = i * j;
		}
	}

	// B is the identity matrix
	for(int i = 0; i < M; i++) {
		for(int j = 0; j < K; j++) {
			B[i * M + j] = (i == j) ? 1 : 0;
		}
	}

	// Load kernel source
	const char kernel_path[] = "./matrix_mul.cl";
	size_t kernel_size = 0;
	char* kernel_source = load_kernel_source(kernel_path, &kernel_size);
	if(kernel_source == NULL) return EXIT_FAILURE;

	cl_int err;

	// Platform & device
	cl_platform_id platform;
	cl_device_id device;
	err = clGetPlatformIDs(1, &platform, NULL);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

	// Context & queue
	cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, &err);

	// Buffers
	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_WRITE, N * M * sizeof(VALUE), NULL, &err);
	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_WRITE, M * K * sizeof(VALUE), NULL, &err);
	cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * K * sizeof(VALUE), NULL, &err);
	cl_mem dm = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, &err);
	cl_mem dk = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, &err);

	// Program from file source
	const char* sources[] = {kernel_source};
	const size_t lengths[] = {kernel_size};
	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

	if(err != CL_SUCCESS) {
		// Print build log
		size_t len;
		char buffer[2048];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		fprintf(stderr, "Build error:\n%s\n", buffer);
		return EXIT_FAILURE;
	}

	// Kernel
	cl_kernel kernel = clCreateKernel(program, "matrix_mul", &err);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
	clSetKernelArg(kernel, 3, sizeof(cl_mem), &dm);
	clSetKernelArg(kernel, 4, sizeof(cl_mem), &dk);

	const double start_time = omp_get_wtime();

	size_t global_work_size[2] = { (size_t)N, (size_t)K };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);

	// Read back result
	clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, N * K * sizeof(VALUE), C, 0, NULL, NULL);

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;


	// verify result
	int success = 1;
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < MIN(M, K); j++) {
			if(A[i * N + j] != C[i * N + j]) { success = 0; }
		}
		for(int j = MIN(M, K); j < MAX(M, K); j++) {
			if(C[i * N + j] != 0) { success = 0; }
		}
	}

	// print verification result
	printf("Verification: %4s\n", (success) ? "OK" : "ERR");
	printf("Time: %9.3f ms\n", elapsed_ms);

	// Cleanup
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);
	clReleaseMemObject(dm);
	clReleaseMemObject(dk);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(kernel_source);

	return EXIT_SUCCESS;
}
