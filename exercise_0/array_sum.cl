__kernel void init_arrays(__global int* A, __global int* B) {
	int i = get_global_id(0);
	A[i] = i + 42;
	B[i] = -i;
}

__kernel void add_arrays(__global int* A, __global int* B, __global int* C) {
	int i = get_global_id(0);
	C[i] = A[i] + B[i];
}