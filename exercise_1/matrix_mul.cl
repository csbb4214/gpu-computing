__kernel void matrix_mul_double(const __global double* A, const __global double* B, __global double* C, const int M, const int K) {
	int row = get_global_id(0); // Row index in C
	int col = get_global_id(1); // Column index in C

	double sum = 0.0f;
	for(int k = 0; k < M; ++k) {
		double a = A[row * M + k]; // A[row][k]
		double b = B[k * K + col]; // B[k][col]
		sum += a * b;
	}

	C[row * K + col] = sum; // C[row][col] = sum
}

__kernel void matrix_mul_float(const __global float* A, const __global float* B, __global float* C, const int M, const int K) {
	int row = get_global_id(0); // Row index in C
	int col = get_global_id(1); // Column index in C

	float sum = 0.0f;
	for(int k = 0; k < M; ++k) {
		float a = A[row * M + k]; // A[row][k]
		float b = B[k * K + col]; // B[k][col]
		sum += a * b;
	}

	C[row * K + col] = sum; // C[row][col] = sum
}