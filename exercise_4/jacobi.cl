#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
/* cl_khr_fp64 is not available; double-precision kernel will be excluded. */
#endif

#ifdef cl_khr_fp64
__kernel void jacobi_step_double(const __global double* u, __global double* tmp, const __global double* f, const double factor) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	int N = get_global_size(0); // assume square

	if(i == 0 || i == N - 1 || j == 0 || j == N - 1) return;

	int idx = i * N + j;
	int up = (i - 1) * N + j;
	int down = (i + 1) * N + j;
	int left = i * N + (j - 1);
	int right = i * N + (j + 1);

	tmp[idx] = 0.25 * (u[up] + u[down] + u[left] + u[right] - factor * f[idx]);
}
#endif

__kernel void jacobi_step_float(const __global float* u, __global float* tmp, const __global float* f, const float factor) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	int N = get_global_size(0); // assume square

	if(i == 0 || i == N - 1 || j == 0 || j == N - 1) return;

	int idx = i * N + j;
	int up = (i - 1) * N + j;
	int down = (i + 1) * N + j;
	int left = i * N + (j - 1);
	int right = i * N + (j + 1);

	tmp[idx] = 0.25f * (u[up] + u[down] + u[left] + u[right] - factor * f[idx]);
}
