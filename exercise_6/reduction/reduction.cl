#ifdef FLOAT
typedef float VALUE;
#define ZERO (0.0f)
#else
typedef int VALUE;
#define ZERO (0)
#endif

__kernel void parallel_reduction(__global VALUE* buffer, __local VALUE* scratch, const int length, __global VALUE* result) {
	int global_index = get_global_id(0);
	int local_index = get_local_id(0);

	// Load data into local memory
	if(global_index < length) {
		scratch[local_index] = buffer[global_index];
	} else {
		scratch[local_index] = ZERO;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Tree reduction
	for(int offset = 1; offset < get_local_size(0); offset <<= 1) {
		int mask = (offset << 1) - 1;
		if((local_index & mask) == 0) {
			VALUE other = scratch[local_index + offset];
			VALUE mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write group result
	if(local_index == 0) result[get_group_id(0)] = scratch[0];
}


__kernel void multistage_reduction(__global VALUE* buffer, __local VALUE* scratch, const int length, __global VALUE* result) {
	int global_index = get_global_id(0);

	// Loop-sequential accumulation
	VALUE accumulator = ZERO;
	while(global_index < length) {
		accumulator += buffer[global_index];
		global_index += get_global_size(0);
	}

	// Write accumulator to local memory
	int local_index = get_local_id(0);
	scratch[local_index] = accumulator;
	barrier(CLK_LOCAL_MEM_FENCE);

	// Tree reduction
	for(int offset = get_local_size(0) / 2; offset > 0; offset >>= 1) {
		if(local_index < offset) {
			VALUE other = scratch[local_index + offset];
			VALUE mine = scratch[local_index];
			scratch[local_index] = mine + other;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(local_index == 0) result[get_group_id(0)] = scratch[0];
}
