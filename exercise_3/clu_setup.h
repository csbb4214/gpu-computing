#ifndef CLU_SETUP_H
#define CLU_SETUP_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

typedef struct {
	cl_platform_id platform_id;
	cl_device_id device_id;
	cl_context context;
	cl_command_queue command_queue;
} clu_env;

/**
 * Initialize OpenCL environment (platform, device, context, command queue).
 * Returns 0 on success, non-zero on failure.
 */
int clu_initialize(clu_env* env);

/**
 * Check if the device supports double precision floating-point.
 * Returns 1 if supported, 0 otherwise.
 */
int clu_check_double_support(cl_device_id device_id);

/**
 * Load kernel source from file.
 * Returns allocated string (caller must free) or NULL on failure.
 */
char* clu_load_kernel_source(const char* path, size_t* out_size);

/**
 * Create and build an OpenCL program from source.
 * Returns program on success, NULL on failure (prints build log on error).
 */
cl_program clu_create_program(cl_context context, cl_device_id device_id, const char* source, size_t source_size, const char* build_options);

/**
 * Release all OpenCL resources in the environment.
 */
void clu_release(clu_env* env);


#endif // CLU_SETUP_H