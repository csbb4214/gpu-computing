#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// COLS_PER_THREAD is provided by the host via -DCOLS_PER_THREAD=...
// Default: 2, if nothing is defined.
#ifndef COLS_PER_THREAD
#define COLS_PER_THREAD 2
#endif

#if COLS_PER_THREAD < 1 || COLS_PER_THREAD > 4
#error "COLS_PER_THREAD must be between 1 and 4"
#endif

// ====================== double kernel (optional) ======================
#ifdef cl_khr_fp64
__kernel void matrix_mul_double_2cols(
    const __global double* A,
    const __global double* B,
    __global double* C,
    const int N,
    const int M,
    const int K)
{
    int row       = get_global_id(0);    // row index in C
    int col_group = get_global_id(1);    // column-group index
    int col0      = col_group * COLS_PER_THREAD;

    if (row >= N || col0 >= K) {
        return;
    }

#if COLS_PER_THREAD >= 1
    double sum0 = 0.0;
#endif
#if COLS_PER_THREAD >= 2
    double sum1 = 0.0;
#endif
#if COLS_PER_THREAD >= 3
    double sum2 = 0.0;
#endif
#if COLS_PER_THREAD >= 4
    double sum3 = 0.0;
#endif

    // unroll inner loop a bit for better ILP
    #pragma unroll 4
    for (int k = 0; k < M; ++k) {
        double a  = A[row * M + k];
        int idx   = k * K + col0;

#if COLS_PER_THREAD >= 1
        double b0 = (col0 + 0 < K) ? B[idx + 0] : 0.0;
        sum0 += a * b0;
#endif
#if COLS_PER_THREAD >= 2
        double b1 = (col0 + 1 < K) ? B[idx + 1] : 0.0;
        sum1 += a * b1;
#endif
#if COLS_PER_THREAD >= 3
        double b2 = (col0 + 2 < K) ? B[idx + 2] : 0.0;
        sum2 += a * b2;
#endif
#if COLS_PER_THREAD >= 4
        double b3 = (col0 + 3 < K) ? B[idx + 3] : 0.0;
        sum3 += a * b3;
#endif
    }

#if COLS_PER_THREAD >= 1
    if (col0 + 0 < K) C[row * K + (col0 + 0)] = sum0;
#endif
#if COLS_PER_THREAD >= 2
    if (col0 + 1 < K) C[row * K + (col0 + 1)] = sum1;
#endif
#if COLS_PER_THREAD >= 3
    if (col0 + 2 < K) C[row * K + (col0 + 2)] = sum2;
#endif
#if COLS_PER_THREAD >= 4
    if (col0 + 3 < K) C[row * K + (col0 + 3)] = sum3;
#endif
}
#endif // cl_khr_fp64


// ====================== float kernel ======================
__kernel void matrix_mul_float_2cols(
    const __global float* A,
    const __global float* B,
    __global float* C,
    const int N,
    const int M,
    const int K)
{
    int row       = get_global_id(0);    // row index in C
    int col_group = get_global_id(1);    // column-group index
    int col0      = col_group * COLS_PER_THREAD;

    if (row >= N || col0 >= K) {
        return;
    }

#if COLS_PER_THREAD >= 1
    float sum0 = 0.0f;
#endif
#if COLS_PER_THREAD >= 2
    float sum1 = 0.0f;
#endif
#if COLS_PER_THREAD >= 3
    float sum2 = 0.0f;
#endif
#if COLS_PER_THREAD >= 4
    float sum3 = 0.0f;
#endif

    // unroll inner loop a bit for better ILP
    #pragma unroll 4
    for (int k = 0; k < M; ++k) {
        float a  = A[row * M + k];
        int idx  = k * K + col0;

#if COLS_PER_THREAD >= 1
        float b0 = (col0 + 0 < K) ? B[idx + 0] : 0.0f;
        sum0 += a * b0;
#endif
#if COLS_PER_THREAD >= 2
        float b1 = (col0 + 1 < K) ? B[idx + 1] : 0.0f;
        sum1 += a * b1;
#endif
#if COLS_PER_THREAD >= 3
        float b2 = (col0 + 2 < K) ? B[idx + 2] : 0.0f;
        sum2 += a * b2;
#endif
#if COLS_PER_THREAD >= 4
        float b3 = (col0 + 3 < K) ? B[idx + 3] : 0.0f;
        sum3 += a * b3;
#endif
    }

#if COLS_PER_THREAD >= 1
    if (col0 + 0 < K) C[row * K + (col0 + 0)] = sum0;
#endif
#if COLS_PER_THREAD >= 2
    if (col0 + 1 < K) C[row * K + (col0 + 1)] = sum1;
#endif
#if COLS_PER_THREAD >= 3
    if (col0 + 2 < K) C[row * K + (col0 + 2)] = sum2;
#endif
#if COLS_PER_THREAD >= 4
    if (col0 + 3 < K) C[row * K + (col0 + 3)] = sum3;
#endif
}

