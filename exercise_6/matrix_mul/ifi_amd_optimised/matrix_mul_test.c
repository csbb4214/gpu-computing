#include "clu_errcheck.h"
#include "clu_setup.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 1000
#endif

#define M N
#define K N

#if defined(USE_DOUBLE)
#define VALUE double
#define KERNEL_NAME "matrix_mul_double_2cols"
#else
#define VALUE float
#define KERNEL_NAME "matrix_mul_float_2cols"
#endif

// Host-Matrizen
VALUE A[N * M];
VALUE B[M * K];
VALUE C[N * K];

int main(void) {
	// ====== Host-Matrizen initialisieren ======
	for(size_t i = 0; i < (size_t)N; i++) {
		for(size_t j = 0; j < (size_t)M; j++) {
			A[i * (size_t)M + j] = (VALUE)1.0;
		}
	}

	for(size_t i = 0; i < (size_t)M; i++) {
		for(size_t j = 0; j < (size_t)K; j++) {
			B[i * (size_t)K + j] = (VALUE)1.0;
		}
	}

	for(size_t i = 0; i < (size_t)N; i++) {
		for(size_t j = 0; j < (size_t)K; j++) {
			C[i * (size_t)K + j] = (VALUE)0.0;
		}
	}

	// ====== OpenCL-Initialisierung ======
	clu_env env;
	cl_queue_properties queue_properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

	if(clu_initialize(&env, queue_properties) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		return EXIT_FAILURE;
	}

	// Gerätename und Max-Work-Group-Size ausgeben
	size_t max_wg_size = 0;
	{
		char device_name[256] = {0};
		clGetDeviceInfo(env.device_id, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
		printf("Using OpenCL device: %s\n", device_name);

		clGetDeviceInfo(env.device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg_size), &max_wg_size, NULL);
		printf("Max work-group size: %zu\n", max_wg_size);
	}

#if defined(USE_DOUBLE)
	if(!clu_check_double_support(env.device_id)) {
		fprintf(stderr, "Device does not support double precision.\n");
		clu_release(&env);
		return EXIT_FAILURE;
	}
#endif

	// ====== Kernel-Source laden ======
	const char kernel_path[] = "./matrix_mul.cl";
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source(kernel_path, &source_size);
	if(source_str == NULL) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	// ====== Buffer anlegen (einmal) ======
	cl_int err = CL_SUCCESS;

	cl_mem bufA = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * (size_t)N * (size_t)M, A, &err);
	CLU_ERRCHECK(err);

	cl_mem bufB = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(VALUE) * (size_t)M * (size_t)K, B, &err);
	CLU_ERRCHECK(err);

	cl_mem bufC = clCreateBuffer(env.context, CL_MEM_WRITE_ONLY, sizeof(VALUE) * (size_t)N * (size_t)K, NULL, &err);
	CLU_ERRCHECK(err);

	const cl_int N_arg = (cl_int)N;
	const cl_int M_arg = (cl_int)M;
	const cl_int K_arg = (cl_int)K;

	// ====== Kandidaten für Tuning ======
	// Du kannst diese Arrays jederzeit erweitern/verändern
	const int cols_candidates[] = {1, 2, 4};
	const int tile_x_candidates[] = {4, 8, 16, 32};
	const int tile_y_candidates[] = {1, 2, 4, 8, 16, 32};

	const size_t num_cols_candidates = sizeof(cols_candidates) / sizeof(cols_candidates[0]);
	const size_t num_tx_candidates = sizeof(tile_x_candidates) / sizeof(tile_x_candidates[0]);
	const size_t num_ty_candidates = sizeof(tile_y_candidates) / sizeof(tile_y_candidates[0]);

	// Anzahl der Wiederholungen pro Kombination
	const int runs_per_combo = 3;

	double best_time_ms = 1.0e30;
	int best_cols = 0;
	int best_tx = 0;
	int best_ty = 0;

	printf("Starting parameter search (each combo %d runs)...\n", runs_per_combo);

	// ====== Parameter-Suche ======
	for(size_t ci = 0; ci < num_cols_candidates; ++ci) {
		for(size_t xi = 0; xi < num_tx_candidates; ++xi) {
			for(size_t yi = 0; yi < num_ty_candidates; ++yi) {
				int cols_per_thread = cols_candidates[ci];
				int tile_x = tile_x_candidates[xi];
				int tile_y = tile_y_candidates[yi];

				size_t wg_size = (size_t)tile_x * (size_t)tile_y;
				if(wg_size > max_wg_size) {
					// Work-Group ist größer als vom Gerät erlaubt
					continue;
				}

				// Build-Optionen mit aktuellen Parametern
				char build_options[512];

#if defined(USE_DOUBLE)
				snprintf(build_options, sizeof(build_options),
				    "-DUSE_DOUBLE=1 "
				    "-DCOLS_PER_THREAD=%d -DTILE_X=%d -DTILE_Y=%d "
				    "-cl-mad-enable -cl-fast-relaxed-math",
				    cols_per_thread, tile_x, tile_y);
#else
				snprintf(build_options, sizeof(build_options),
				    "-DCOLS_PER_THREAD=%d -DTILE_X=%d -DTILE_Y=%d "
				    "-cl-mad-enable -cl-fast-relaxed-math",
				    cols_per_thread, tile_x, tile_y);
#endif

				// Programm für diese Parameter bauen
				cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, build_options);

				if(program == NULL) {
					fprintf(stderr, "Build failed for cols=%d, TILE_X=%d, TILE_Y=%d\n", cols_per_thread, tile_x, tile_y);
					continue;
				}

				cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
				if(err != CL_SUCCESS) {
					fprintf(stderr, "Failed to create kernel for cols=%d, TILE_X=%d, TILE_Y=%d\n", cols_per_thread, tile_x, tile_y);
					clReleaseProgram(program);
					continue;
				}

				// Kernel-Argumente setzen
				err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
				CLU_ERRCHECK(err);
				err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
				CLU_ERRCHECK(err);
				err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
				CLU_ERRCHECK(err);
				err = clSetKernelArg(kernel, 3, sizeof(cl_int), &N_arg);
				CLU_ERRCHECK(err);
				err = clSetKernelArg(kernel, 4, sizeof(cl_int), &M_arg);
				CLU_ERRCHECK(err);
				err = clSetKernelArg(kernel, 5, sizeof(cl_int), &K_arg);
				CLU_ERRCHECK(err);

				const size_t num_col_groups = ((size_t)K + (size_t)cols_per_thread - 1) / (size_t)cols_per_thread;

				size_t global_work_size[2];
				global_work_size[0] = ((size_t)N + (size_t)tile_x - 1) / (size_t)tile_x * (size_t)tile_x;
				global_work_size[1] = (num_col_groups + (size_t)tile_y - 1) / (size_t)tile_y * (size_t)tile_y;

				const size_t local_work_size[2] = {(size_t)tile_x, (size_t)tile_y};

				// Mehrfach-Läufe pro Kombination
				double sum_ms = 0.0;

				for(int run = 0; run < runs_per_combo; ++run) {
					double start_time = omp_get_wtime();

					err = clEnqueueNDRangeKernel(env.command_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
					CLU_ERRCHECK(err);

					err = clFinish(env.command_queue);
					CLU_ERRCHECK(err);

					double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;
					sum_ms += elapsed_ms;
				}

				double avg_ms = sum_ms / (double)runs_per_combo;

				printf(
				    "cols=%d, TILE_X=%d, TILE_Y=%d, WG_SIZE=%zu -> avg %.3f ms (%d runs)\n", cols_per_thread, tile_x, tile_y, wg_size, avg_ms, runs_per_combo);

				if(avg_ms < best_time_ms) {
					best_time_ms = avg_ms;
					best_cols = cols_per_thread;
					best_tx = tile_x;
					best_ty = tile_y;
				}

				clReleaseKernel(kernel);
				clReleaseProgram(program);
			}
		}
	}

	if(best_cols == 0) {
		fprintf(stderr, "No valid parameter combination found.\n");
		clReleaseMemObject(bufC);
		clReleaseMemObject(bufB);
		clReleaseMemObject(bufA);
		free(source_str);
		clu_release(&env);
		return EXIT_FAILURE;
	}

	printf("\nBEST COMBINATION (avg over %d runs):\n", runs_per_combo);
	printf("  COLS_PER_THREAD = %d\n", best_cols);
	printf("  TILE_X          = %d\n", best_tx);
	printf("  TILE_Y          = %d\n", best_ty);
	printf("  WG_SIZE         = %zu\n", (size_t)best_tx * (size_t)best_ty);
	printf("  AVG TIME        = %.3f ms\n\n", best_time_ms);

	printf("You can now hard-code:\n");
	printf("#define COLS_PER_THREAD %d\n", best_cols);
	printf("#define TILE_X %d\n", best_tx);
	printf("#define TILE_Y %d\n\n", best_ty);

	// ====== Optional: Noch einmal mit bester Kombination ausführen & C[0,0] prüfen ======
	char best_build_options[512];

#if defined(USE_DOUBLE)
	snprintf(best_build_options, sizeof(best_build_options),
	    "-DUSE_DOUBLE=1 "
	    "-DCOLS_PER_THREAD=%d -DTILE_X=%d -DTILE_Y=%d "
	    "-cl-mad-enable -cl-fast-relaxed-math",
	    best_cols, best_tx, best_ty);
#else
	snprintf(best_build_options, sizeof(best_build_options),
	    "-DCOLS_PER_THREAD=%d -DTILE_X=%d -DTILE_Y=%d "
	    "-cl-mad-enable -cl-fast-relaxed-math",
	    best_cols, best_tx, best_ty);
#endif

	cl_program best_program = clu_create_program(env.context, env.device_id, source_str, source_size, best_build_options);
	if(best_program == NULL) {
		fprintf(stderr, "Failed to build best program.\n");
		clReleaseMemObject(bufC);
		clReleaseMemObject(bufB);
		clReleaseMemObject(bufA);
		free(source_str);
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_kernel best_kernel = clCreateKernel(best_program, KERNEL_NAME, &err);
	CLU_ERRCHECK(err);

	err = clSetKernelArg(best_kernel, 0, sizeof(cl_mem), &bufA);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(best_kernel, 1, sizeof(cl_mem), &bufB);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(best_kernel, 2, sizeof(cl_mem), &bufC);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(best_kernel, 3, sizeof(cl_int), &N_arg);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(best_kernel, 4, sizeof(cl_int), &M_arg);
	CLU_ERRCHECK(err);
	err = clSetKernelArg(best_kernel, 5, sizeof(cl_int), &K_arg);
	CLU_ERRCHECK(err);

	const size_t best_num_col_groups = ((size_t)K + (size_t)best_cols - 1) / (size_t)best_cols;

	size_t best_global_work_size[2];
	best_global_work_size[0] = ((size_t)N + (size_t)best_tx - 1) / (size_t)best_tx * (size_t)best_tx;
	best_global_work_size[1] = (best_num_col_groups + (size_t)best_ty - 1) / (size_t)best_ty * (size_t)best_ty;

	const size_t best_local_work_size[2] = {(size_t)best_tx, (size_t)best_ty};

	double start_time = omp_get_wtime();

	err = clEnqueueNDRangeKernel(env.command_queue, best_kernel, 2, NULL, best_global_work_size, best_local_work_size, 0, NULL, NULL);
	CLU_ERRCHECK(err);

	err = clFinish(env.command_queue);
	CLU_ERRCHECK(err);

	double final_ms = (omp_get_wtime() - start_time) * 1000.0;

	err = clEnqueueReadBuffer(env.command_queue, bufC, CL_TRUE, 0, sizeof(VALUE) * (size_t)N * (size_t)K, C, 0, NULL, NULL);
	CLU_ERRCHECK(err);

	printf("Final run with best params: time = %.3f ms, C[0,0] = %f\n", final_ms, (double)C[0]);

	// ====== Cleanup ======
	clReleaseKernel(best_kernel);
	clReleaseProgram(best_program);

	clReleaseMemObject(bufC);
	clReleaseMemObject(bufB);
	clReleaseMemObject(bufA);

	free(source_str);
	clu_release(&env);

	return EXIT_SUCCESS;
}
