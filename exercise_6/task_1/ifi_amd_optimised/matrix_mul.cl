#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// each work-item computes this many columns
#define COLS_PER_THREAD 2

#ifdef cl_khr_fp64
__kernel void matrix_mul_double_2cols(
    const __global double* A,
    const __global double* B,
    __global double* C,
    const int N,
    const int M,
    const int K)
{
    int row = get_global_id(0);      // row index in C
    int col_group = get_global_id(1); // column group index
    int col0 = col_group * COLS_PER_THREAD;

    if (row >= N || col0 >= K) {
        return;
    }

    double sum0 = 0.0;
    double sum1 = 0.0;

    // unroll inner loop a bit for better ILP
    #pragma unroll 4
    for (int k = 0; k < M; ++k) {
        double a = A[row * M + k];
        int idx = k * K + col0;

        double b0 = B[idx];
        double b1 = (col0 + 1 < K) ? B[idx + 1] : 0.0;

        sum0 += a * b0;
        sum1 += a * b1;
    }

    C[row * K + col0] = sum0;
    if (col0 + 1 < K) {
        C[row * K + (col0 + 1)] = sum1;
    }
}
#endif

__kernel void matrix_mul_float_2cols(
    const __global float* A,
    const __global float* B,
    __global float* C,
    const int N,
    const int M,
    const int K)
{
    int row = get_global_id(0);      // row index in C
    int col_group = get_global_id(1); // column group index
    int col0 = col_group * COLS_PER_THREAD;

    if (row >= N || col0 >= K) {
        return;
    }

    float sum0 = 0.0f;
    float sum1 = 0.0f;

    // unroll inner loop a bit for better ILP
    #pragma unroll 4
    for (int k = 0; k < M; ++k) {
        float a = A[row * M + k];
        int idx = k * K + col0;

        float b0 = B[idx];
        float b1 = (col0 + 1 < K) ? B[idx + 1] : 0.0f;

        sum0 += a * b0;
        sum1 += a * b1;
    }

    C[row * K + col0] = sum0;
    if (col0 + 1 < K) {
        C[row * K + (col0 + 1)] = sum1;
    }
}
