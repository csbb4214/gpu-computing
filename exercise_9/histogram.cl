__kernel void histogram(const __global int* ages, __global int* C, const int N) {
	int gid = get_global_id(0);

	if(gid < N) {
		int age = ages[gid];
		atomic_inc(&C[age]);
	}
}
