#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define TSX 16
#define TSY 16

#ifdef cl_khr_fp64
__kernel void matrix_mul_tiled_double(
    __global const double* A,
    __global const double* B,
    __global double* C,
    const int M,
    const int K)
{
    __local double Asub[TSX][TSY];
    __local double Bsub[TSX][TSY];

    int row = get_global_id(0);
    int col = get_global_id(1);
    double sum = 0.0;

    for(int t = 0; t < (M + TSX - 1) / TSX; t++) {
        int tiledColA = t*TSX + get_local_id(1);
        int tiledRowB = t*TSX + get_local_id(0);

        Asub[get_local_id(0)][get_local_id(1)] = (row < M && tiledColA < M) ? A[row*M + tiledColA] : 0.0;
        Bsub[get_local_id(0)][get_local_id(1)] = (tiledRowB < M && col < K) ? B[tiledRowB*K + col] : 0.0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TSY; k++)
            sum += Asub[get_local_id(0)][k] * Bsub[k][get_local_id(1)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < K)
        C[row*K + col] = sum;
}
#endif

__kernel void matrix_mul_tiled_float(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int K)
{
    __local float Asub[TSX][TSY];
    __local float Bsub[TSX][TSY];

    int row = get_global_id(0);
    int col = get_global_id(1);
    float sum = 0.0f;

    for(int t = 0; t < (M + TSX - 1) / TSX; t++) {
        int tiledColA = t*TSX + get_local_id(1);
        int tiledRowB = t*TSX + get_local_id(0);

        Asub[get_local_id(0)][get_local_id(1)] = (row < M && tiledColA < M) ? A[row*M + tiledColA] : 0.0f;
        Bsub[get_local_id(0)][get_local_id(1)] = (tiledRowB < M && col < K) ? B[tiledRowB*K + col] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for(int k = 0; k < TSY; k++)
            sum += Asub[get_local_id(0)][k] * Bsub[k][get_local_id(1)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < K)
        C[row*K + col] = sum;
}

