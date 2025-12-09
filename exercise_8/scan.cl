#ifdef FLOAT
#define VALUE float
#define ZERO 0.0f
#else
#define VALUE int
#define ZERO 0
#endif

// Phase 1: Local scan within each work-group
__kernel void hillis_steele_scan(__global VALUE* g_odata, __global VALUE* g_idata, int n, __local VALUE* temp, __global VALUE* block_sums) {
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int group_id = get_group_id(0);
	int group_size = get_local_size(0);

	// Double buffering indices
	int pout = 0;
	int pin = 1;

	// Load input into local memory
	if(global_id < n) {
		temp[local_id] = g_idata[global_id];
	} else {
		temp[local_id] = ZERO;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Hillis & Steele algorithm within work-group
	for(int offset = 1; offset < group_size; offset <<= 1) {
		// Swap double buffer indices
		pout = 1 - pout;
		pin = 1 - pout;

		if(local_id >= offset) {
			temp[pout * group_size + local_id] = temp[pin * group_size + local_id] + temp[pin * group_size + local_id - offset];
		} else {
			temp[pout * group_size + local_id] = temp[pin * group_size + local_id];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// Write output
	if(global_id < n) { g_odata[global_id] = temp[pout * group_size + local_id]; }

	// Last thread in work-group writes the block sum
	if(local_id == group_size - 1) { block_sums[group_id] = temp[pout * group_size + local_id]; }
}

// Phase 2: Add block sums to all elements
__kernel void add_block_sums(__global VALUE* g_data, __global VALUE* block_sums, int n) {
	int global_id = get_global_id(0);
	int group_id = get_group_id(0);

	if(global_id < n && group_id > 0) { g_data[global_id] += block_sums[group_id - 1]; }
}