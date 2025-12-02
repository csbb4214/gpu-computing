#include "clu_setup.h"
#include "clu_errcheck.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int clu_initialize(clu_env* env, cl_queue_properties* queue_properties) {
	if(env == NULL) { return -1; }

	memset(env, 0, sizeof(clu_env));

	cl_int err = CL_SUCCESS;

	cl_uint platform_count = 0;
	err = clGetPlatformIDs(1, &env->platform_id, &platform_count);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		return -1;
	}

	cl_uint device_count = 0;
	err = clGetDeviceIDs(env->platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &env->device_id, &device_count);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		return -1;
	}

	env->context = clCreateContext(NULL, 1, &env->device_id, NULL, NULL, &err);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		return -1;
	}

	env->command_queue = clCreateCommandQueueWithProperties(env->context, env->device_id, queue_properties, &err);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		clReleaseContext(env->context);
		env->context = NULL;
		return -1;
	}

	return 0;
}

int clu_check_double_support(cl_device_id device_id) {
	cl_device_fp_config fp64_config = 0;
	cl_int err = clGetDeviceInfo(device_id, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &fp64_config, NULL);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		return 0;
	}
	return fp64_config != 0;
}

char* clu_load_kernel_source(const char* path, size_t* out_size) {
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

	const size_t read = fread(buffer, 1U, (size_t)length, file);
	if(read != (size_t)length) {
		fprintf(stderr, "Failed to read kernel source '%s'.\n", path);
		free(buffer);
		fclose(file);
		return NULL;
	}

	buffer[length] = '\0';
	if(out_size != NULL) { *out_size = read; }

	fclose(file);
	return buffer;
}

cl_program clu_create_program(cl_context context, cl_device_id device_id, const char* source, size_t source_size, const char* build_options) {
	cl_int err = CL_SUCCESS;

	const char* sources[] = {source};
	const size_t lengths[] = {source_size};

	cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		return NULL;
	}

	err = clBuildProgram(program, 1, &device_id, build_options, NULL, NULL);
	cluPrintProgramBuildLog(program, device_id);
	if(err != CL_SUCCESS) {
		fprintf(stderr, "OpenCL error: %s (%d)\n", cluErrorString(err), err);
		clReleaseProgram(program);
		return NULL;
	}

	return program;
}

void clu_release(clu_env* env) {
	if(env == NULL) { return; }

	if(env->command_queue != NULL) { clReleaseCommandQueue(env->command_queue); }
	if(env->context != NULL) { clReleaseContext(env->context); }

	memset(env, 0, sizeof(clu_env));
}