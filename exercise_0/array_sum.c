#include <CL/cl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int main(int argc, char** argv) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s <array_size>\n", argv[0]);
		return EXIT_FAILURE;
	}

	size_t N = atoi(argv[1]);
	size_t bytes = N * sizeof(int);
	int* C = (int*)malloc(bytes);
	if(!C) {
		fprintf(stderr, "Failed to allocate host memory\n");
		return EXIT_FAILURE;
	}

	// Load kernel source
	const char kernel_path[] = "./array_sum.cl";
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
	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, &err);
	cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

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

	// Kernel 1
	cl_kernel k_init = clCreateKernel(program, "init_arrays", &err);
	clSetKernelArg(k_init, 0, sizeof(cl_mem), &dA);
	clSetKernelArg(k_init, 1, sizeof(cl_mem), &dB);

	size_t globalSize = N;
	clEnqueueNDRangeKernel(queue, k_init, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

	// Kernel 2
	cl_kernel k_add = clCreateKernel(program, "add_arrays", &err);
	clSetKernelArg(k_add, 0, sizeof(cl_mem), &dA);
	clSetKernelArg(k_add, 1, sizeof(cl_mem), &dB);
	clSetKernelArg(k_add, 2, sizeof(cl_mem), &dC);

	clEnqueueNDRangeKernel(queue, k_add, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

	// Read back result
	clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, bytes, C, 0, NULL, NULL);

	// Verification
	int errors = 0;
	for(size_t i = 0; i < N; i++) {
		int expected = (i + 42) + (-i);
		if(C[i] != expected) {
			printf("Fehler bei Index %zu: %d != %d\n", i, C[i], expected);
			if(++errors > 10) break;
		}
	}

	if(errors == 0)
		printf("Alle %zu Elemente korrekt!\n", N);
	else
		printf("%d Fehler gefunden.\n", errors);

	// Cleanup
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);
	clReleaseKernel(k_init);
	clReleaseKernel(k_add);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
	free(C);
	free(kernel_source);

	return EXIT_SUCCESS;
}
