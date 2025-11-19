__kernel void parallel_reduction(__global float* buffer, __local float* scratch, __const int length, __global float* result) {
	int global_index = get_global_id(0);
	int local_index = get_local_id(0); // Load data into local memory
	if(global_index < length)
		scratch[local_index] = buffer[global_index];
	else {
		// 0 is the identity element for the sum operation
		scratch[local_index] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE); // Finish initializing local memory
	for(int offset = 1; offset < get_local_size(0); offset <<= 1) {
		int mask = (offset << 1) - 1;   // e.g. offset 8 → mask 0000’0111
		if((local_index & mask) == 0) { // use all wis not masked out
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if(local_index == 0) { result[get_group_id(0)] = scratch[0]; }
}

__kernel void multistate_reduction(__global float* buffer, __local float* scratch, __const int length, __global float* result) {
	int global_index = get_global_id(0);
	float accumulator = 0;
	while(global_index < length) { // loop sequentially over chunks of input vector
		accumulator += buffer[global_index];
		global_index += get_global_size(0);
	}

	// Perform parallel reduction
	int local_index = get_local_id(0);
	scratch[local_index] = accumulator;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int offset = get_local_size(0) / 2; offset > 0; offset = offset / 2) {
		if(local_index < offset) {
			float other = scratch[local_index + offset];
			float mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(local_index == 0) { result[get_group_id(0)] = scratch[0]; }
}
