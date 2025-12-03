#include "clu_errcheck.h"
#include "clu_setup.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wmisleading-indentation"
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_COMPONENTS 4
#define WORKGROUP_SIZE 256

typedef struct {
	stbi_uc min;
	stbi_uc max;
	unsigned long long sum;
} ComponentStats;

int main(int argc, char** argv) {
	if(argc != 3) {
		printf("Usage: auto_levels [inputfile] [outputfile]\nExample: auto_levels test.png test_adjusted.png\n");
		return -1;
	}

	int width, height, components;
	stbi_uc* data = stbi_load(argv[1], &width, &height, &components, 0);

	if(data == NULL) {
		printf("Error loading image %s\n", argv[1]);
		return -1;
	}

	if(components > MAX_COMPONENTS) {
		printf("Too many components: %d (max %d)\n", components, MAX_COMPONENTS);
		stbi_image_free(data);
		return -1;
	}

	const size_t total_pixels = width * height;
	const size_t total_bytes = total_pixels * components;

	// ========== Initialization ==========
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

	if(clu_initialize(&env, queue_properties) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		stbi_image_free(data);
		return EXIT_FAILURE;
	}

	// ========== Load and compile kernel ==========
	const char kernel_path[] = "./auto_levels.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		stbi_image_free(data);
		return EXIT_FAILURE;
	}

	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, NULL);
	free(source_str);

	if(program == NULL) {
		clu_release(&env);
		stbi_image_free(data);
		return EXIT_FAILURE;
	}

	cl_int err = CL_SUCCESS;

	// Create two kernels
	cl_kernel reduce_kernel = clCreateKernel(program, "reduce_stats", &err);
	CLU_ERRCHECK(err);

	cl_kernel adjust_kernel = clCreateKernel(program, "adjust_levels", &err);
	CLU_ERRCHECK(err);

	// Start time (ii)
	const double host_to_host_start = omp_get_wtime();

	// ========== Create buffers ==========
	// Input image buffer
	cl_mem buf_image = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, total_bytes, data, &err);
	CLU_ERRCHECK(err);

	// Output image buffer
	cl_mem buf_output = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, total_bytes, NULL, &err);
	CLU_ERRCHECK(err);

	// ========== KERNEL 1: Reduction to find min/max/sum ==========
	cl_event event_reduction_enqueue, event_reduction_read;

	// Global work size: one work item per pixel
	const size_t global_work_size_reduce = (total_pixels + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
	const size_t local_work_size_reduce = WORKGROUP_SIZE;
	const size_t num_workgroups = global_work_size_reduce / WORKGROUP_SIZE;

	// Buffer for reduction results (min, max, sum for each component)
	// -> 2D array: [num_workgroups * components][3] where 0=min, 1=max, 2=sum
	const size_t stats_buffer_size = num_workgroups * components * 3 * sizeof(unsigned long long);
	cl_mem buf_stats = clCreateBuffer(env.context, CL_MEM_READ_WRITE, stats_buffer_size, NULL, &err);
	CLU_ERRCHECK(err);

	// Set kernel arguments for reduction
	err = clSetKernelArg(reduce_kernel, 0, sizeof(cl_mem), &buf_image);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(reduce_kernel, 1, sizeof(cl_mem), &buf_stats);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(reduce_kernel, 2, sizeof(int), &width);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(reduce_kernel, 3, sizeof(int), &height);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(reduce_kernel, 4, sizeof(int), &components);
	CLU_ERRCHECK(err);

	err =
	    clEnqueueNDRangeKernel(env.command_queue, reduce_kernel, 1, NULL, &global_work_size_reduce, &local_work_size_reduce, 0, NULL, &event_reduction_enqueue);
	CLU_ERRCHECK(err);

	// Read back partial reduction results
	unsigned long long* partial_stats = malloc(stats_buffer_size);
	err = clEnqueueReadBuffer(env.command_queue, buf_stats, CL_TRUE, 0, stats_buffer_size, partial_stats, 0, NULL, &event_reduction_read);
	CLU_ERRCHECK(err);

	// timing kernel 1
	cl_ulong time_reduce_start, time_reduce_end;

	CLU_ERRCHECK(clGetEventProfilingInfo(event_reduction_enqueue, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_reduce_start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(event_reduction_read, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_reduce_end, NULL));

	CLU_ERRCHECK(clReleaseEvent(event_reduction_enqueue));
	CLU_ERRCHECK(clReleaseEvent(event_reduction_read));

	// ========== CPU: Final reduction and factor calculation ==========
	ComponentStats final_stats[MAX_COMPONENTS];
	for(int c = 0; c < components; ++c) {
		final_stats[c].min = 255;
		final_stats[c].max = 0;
		final_stats[c].sum = 0;

		// Combine results from all workgroups
		for(size_t i = 0; i < num_workgroups; ++i) {
			size_t base_idx = (i * components + c) * 3;

			// Min
			stbi_uc wg_min = (stbi_uc)partial_stats[base_idx];
			if(wg_min < final_stats[c].min) final_stats[c].min = wg_min;

			// Max
			stbi_uc wg_max = (stbi_uc)partial_stats[base_idx + 1];
			if(wg_max > final_stats[c].max) final_stats[c].max = wg_max;

			// Sum
			final_stats[c].sum += partial_stats[base_idx + 2];
		}
	}

	free(partial_stats);

	// Calculate averages and scaling factors
	stbi_uc avg_val[MAX_COMPONENTS];
	float min_fac[MAX_COMPONENTS], max_fac[MAX_COMPONENTS];

	for(int c = 0; c < components; ++c) {
		avg_val[c] = (stbi_uc)(final_stats[c].sum / total_pixels);

		if(avg_val[c] != final_stats[c].min) {
			min_fac[c] = (float)avg_val[c] / ((float)avg_val[c] - (float)final_stats[c].min);
		} else {
			min_fac[c] = 1.0f; // Avoid division by zero
		}

		if(final_stats[c].max != avg_val[c]) {
			max_fac[c] = (255.0f - (float)avg_val[c]) / ((float)final_stats[c].max - (float)avg_val[c]);
		} else {
			max_fac[c] = 1.0f; // Avoid division by zero
		}

		// printf("Component %d: %3u/%3u/%3u * %5.2f/%5.2f\n", c, final_stats[c].min, avg_val[c], final_stats[c].max, min_fac[c], max_fac[c]);
	}

	// ========== KERNEL 2: Adjust image levels ==========
	cl_event event_adjust_enqueue, event_adjust_read;

	// Transfer scaling factors to GPU
	cl_mem buf_avg = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, components * sizeof(stbi_uc), avg_val, &err);
	CLU_ERRCHECK(err);

	cl_mem buf_min_fac = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, components * sizeof(float), min_fac, &err);
	CLU_ERRCHECK(err);

	cl_mem buf_max_fac = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, components * sizeof(float), max_fac, &err);
	CLU_ERRCHECK(err);

	// Set kernel arguments for adjustment
	err = clSetKernelArg(adjust_kernel, 0, sizeof(cl_mem), &buf_image);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 1, sizeof(cl_mem), &buf_output);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 2, sizeof(cl_mem), &buf_avg);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 3, sizeof(cl_mem), &buf_min_fac);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 4, sizeof(cl_mem), &buf_max_fac);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 5, sizeof(int), &width);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 6, sizeof(int), &height);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(adjust_kernel, 7, sizeof(int), &components);
	CLU_ERRCHECK(err);

	// Global work size: one work item per pixel
	const size_t global_work_size_adjust[2] = {(size_t)width, (size_t)height};
	const size_t local_work_size_adjust[2] = {16, 16};

	err = clEnqueueNDRangeKernel(env.command_queue, adjust_kernel, 2, NULL, global_work_size_adjust, local_work_size_adjust, 0, NULL, &event_adjust_enqueue);
	CLU_ERRCHECK(err);

	// ========== Read back adjusted image ==========
	err = clEnqueueReadBuffer(env.command_queue, buf_output, CL_TRUE, 0, total_bytes, data, 0, NULL, &event_adjust_read);
	CLU_ERRCHECK(err);

	const double host_to_host_end = omp_get_wtime();

	// timing kernel 2
	cl_ulong time_adjust_start, time_adjust_end;

	CLU_ERRCHECK(clGetEventProfilingInfo(event_adjust_enqueue, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_adjust_start, NULL));
	CLU_ERRCHECK(clGetEventProfilingInfo(event_adjust_read, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_adjust_end, NULL));

	CLU_ERRCHECK(clReleaseEvent(event_adjust_enqueue));
	CLU_ERRCHECK(clReleaseEvent(event_adjust_read));

	// ========== Print timing data ==========
	cl_ulong gpu_cpu_time = time_adjust_end - time_reduce_start;
	cl_ulong gpu_only_time = time_reduce_end - time_reduce_start + time_adjust_end - time_adjust_start;

	const double time_ia = (double)(gpu_cpu_time * 1e-6);
	const double time_ib = (double)(gpu_only_time * 1e-6);
	const double time_ii = (host_to_host_end - host_to_host_start) * 1000.0;

	printf("opencl,%.6f,%.6f,%.6f\n", time_ii, time_ia, time_ib);

	// ========== Save output image ==========
	stbi_write_png(argv[2], width, height, components, data, width * components);

	// ========== Cleanup ==========
	clReleaseMemObject(buf_max_fac);
	clReleaseMemObject(buf_min_fac);
	clReleaseMemObject(buf_avg);
	clReleaseMemObject(buf_output);
	clReleaseMemObject(buf_stats);
	clReleaseMemObject(buf_image);
	clReleaseKernel(adjust_kernel);
	clReleaseKernel(reduce_kernel);
	clReleaseProgram(program);
	clu_release(&env);

	stbi_image_free(data);

	return EXIT_SUCCESS;
}