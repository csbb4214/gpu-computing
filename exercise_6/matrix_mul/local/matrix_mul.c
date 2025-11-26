#include "clu_setup.h"
#include "clu_errcheck.h"

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
#define KERNEL_NAME "matrix_mul_double"
#else
#define VALUE float
#define KERNEL_NAME "matrix_mul_float"
#endif

// host matrices
VALUE A[N * M];
VALUE B[M * K];
VALUE C[N * K];

int main(void) {
    // ========== Initialize host matrices ==========
    for(size_t i = 0; i < (size_t)N; i++) {
        for(size_t j = 0; j < (size_t)M; j++) {
            A[i * M + j] = (VALUE)1.0;
        }
    }

    for(size_t i = 0; i < (size_t)M; i++) {
        for(size_t j = 0; j < (size_t)K; j++) {
            B[i * K + j] = (VALUE)1.0;
        }
    }

    for(size_t i = 0; i < (size_t)N; i++) {
        for(size_t j = 0; j < (size_t)K; j++) {
            C[i * K + j] = (VALUE)0.0;
        }
    }

    // ========== Initialization ==========
    clu_env env;
    cl_queue_properties queue_properties[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };

    if(clu_initialize(&env, queue_properties) != 0) {
        fprintf(stderr, "Failed to initialize OpenCL\n");
        return EXIT_FAILURE;
    }

#if defined(USE_DOUBLE)
    if(!clu_check_double_support(env.device_id)) {
        fprintf(stderr, "Device does not support double precision.\n");
        clu_release(&env);
        return EXIT_FAILURE;
    }
#endif

    // ========== Load and compile kernel ==========
    const char kernel_path[] = "./matrix_mul.cl";
    size_t source_size = 0;
    char* source_str = clu_load_kernel_source(kernel_path, &source_size);
    if(source_str == NULL) {
        clu_release(&env);
        return EXIT_FAILURE;
    }

#if defined(USE_DOUBLE)
    const char* build_options = "-DUSE_DOUBLE=1";
#else
    const char* build_options = "";
#endif

    cl_program program = clu_create_program(
        env.context,
        env.device_id,
        source_str,
        source_size,
        build_options
    );
    free(source_str);

    if(program == NULL) {
        clu_release(&env);
        return EXIT_FAILURE;
    }

    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, KERNEL_NAME, &err);
    CLU_ERRCHECK(err);

    // ========== Create buffers ==========
    cl_mem bufA = clCreateBuffer(
        env.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(VALUE) * N * M,
        A,
        &err
    );
    CLU_ERRCHECK(err);

    cl_mem bufB = clCreateBuffer(
        env.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(VALUE) * M * K,
        B,
        &err
    );
    CLU_ERRCHECK(err);

    cl_mem bufC = clCreateBuffer(
        env.context,
        CL_MEM_WRITE_ONLY,
        sizeof(VALUE) * N * K,
        NULL,
        &err
    );
    CLU_ERRCHECK(err);

    // ========== Set kernel arguments ==========
    const cl_int M_arg = (cl_int)M;
    const cl_int K_arg = (cl_int)K;

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    CLU_ERRCHECK(err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    CLU_ERRCHECK(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    CLU_ERRCHECK(err);

    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &M_arg);
    CLU_ERRCHECK(err);

    err = clSetKernelArg(kernel, 4, sizeof(cl_int), &K_arg);
    CLU_ERRCHECK(err);

    // ========== Launch kernel ==========
    const size_t global_work_size[2] = { (size_t)N, (size_t)K };
    const size_t* local_work_size = NULL;

    const double start_time = omp_get_wtime();

    err = clEnqueueNDRangeKernel(
        env.command_queue,
        kernel,
        2,
        NULL,
        global_work_size,
        local_work_size,
        0,
        NULL,
        NULL
    );
    CLU_ERRCHECK(err);

    err = clFinish(env.command_queue);
    CLU_ERRCHECK(err);

    const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

    // ========== Read back result ==========
    err = clEnqueueReadBuffer(
        env.command_queue,
        bufC,
        CL_TRUE,
        0,
        sizeof(VALUE) * N * K,
        C,
        0,
        NULL,
        NULL
    );
    CLU_ERRCHECK(err);

    printf("C[0,0] = %f, time = %.3f ms\n", (double)C[0], elapsed_ms);

    // ========== Cleanup ==========
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clu_release(&env);

    return EXIT_SUCCESS;
}

