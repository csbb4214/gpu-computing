#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#else
/* cl_khr_fp64 is not available; double-precision kernel will be excluded. */
#endif

#ifdef cl_khr_fp64
__kernel void jacobi_step_double(const __global double* u, __global double* tmp, const __global double* f,
    __local double* tile, // <-- dynamic local memory
    const int pitch,      // i.e. local_size_x + 2 (= local_size_x + borders without padding); = #elements in one row in local memory
    const int N, const double factor) {
	/* get and map indices */
	// get global indices
	int i = get_global_id(0);
	int j = get_global_id(1);
	// get local indices
	int li = get_local_id(0) + 1; // + 1 for border
	int lj = get_local_id(1) + 1; // + 1 for border
	// global index <-> local index
	int idx = i * N + j;
	int local_idx = lj * pitch + li;

	/* Load cells into local memory */
	// Center cell
	tile[local_idx] = u[idx];
	// Left/Right/Up/Down cells
	if(get_local_id(0) == 0) { tile[local_idx - 1] = u[idx - 1]; }
	if(get_local_id(0) == get_local_size(0) - 1) { tile[local_idx + 1] = u[idx + 1]; }
	if(get_local_id(1) == 0) { tile[local_idx - pitch] = u[idx - N]; }
	if(get_local_id(1) == get_local_size(1) - 1) { tile[local_idx + pitch] = u[idx + N]; }

	/* Wait until all threads have filled local memory */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* Compute result using local memory */
	if(i == 0 || i == N - 1 || j == 0 || j == N - 1) return;
	tmp[idx] = 0.25f * (tile[local_idx - 1] + tile[local_idx + 1] + tile[local_idx - pitch] + tile[local_idx + pitch] - factor * f[idx]);
}
#endif

__kernel void jacobi_step_float(
    const __global float* u, __global float* tmp, const __global float* f, __local float* tile, const int pitch, const int N, const float factor) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	int li = get_local_id(0) + 1;
	int lj = get_local_id(1) + 1;

	int idx = i * N + j;
	int local_idx = lj * pitch + li;

	tile[local_idx] = u[idx];

	if(get_local_id(0) == 0) { tile[local_idx - 1] = u[idx - 1]; }
	if(get_local_id(0) == get_local_size(0) - 1) { tile[local_idx + 1] = u[idx + 1]; }
	if(get_local_id(1) == 0) { tile[local_idx - pitch] = u[idx - N]; }
	if(get_local_id(1) == get_local_size(1) - 1) { tile[local_idx + pitch] = u[idx + N]; }

	barrier(CLK_LOCAL_MEM_FENCE);

	if(i == 0 || i == N - 1 || j == 0 || j == N - 1) return;
	tmp[idx] = 0.25f * (tile[local_idx - 1] + tile[local_idx + 1] + tile[local_idx - pitch] + tile[local_idx + pitch] - factor * f[idx]);
}
