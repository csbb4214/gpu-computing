#ifdef FLOAT
#define VALUE float
#define ZERO 0.0f
#else
#define VALUE int
#define ZERO 0
#endif

#ifndef OPT

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

#else

// Improved scan with reduction and down-sweep
__kernel void improved_scan(__global VALUE* g_odata, __global VALUE* g_idata, int n, __local VALUE* temp, __global VALUE* block_sums) {
	int tid = get_local_id(0);
	int gid = get_group_id(0);
	int local_size = get_local_size(0);
	int num_elements = local_size << 1; // process 2 elements per thread

	int ai = tid;
	int bi = tid + local_size;
	int bank_offset_a = ai >> 5;
	int bank_offset_b = bi >> 5;

	// global memory indices
	int g_ai = (gid * num_elements) + ai;
	int g_bi = (gid * num_elements) + bi;

	// load into shared memory with bank offest
	temp[ai + bank_offset_a] = (g_ai < n) ? g_idata[g_ai] : ZERO;
	temp[bi + bank_offset_b] = (g_bi < n) ? g_idata[g_bi] : ZERO;

	// reduce phase
	int offset = 1;
	for(int d = num_elements >> 1; d > 0; d >>= 1) {
		barrier(CLK_LOCAL_MEM_FENCE);
		if(tid < d) {
			int ai_idx = offset * ((tid << 1) + 1) - 1;
			int bi_idx = offset * ((tid << 1) + 2) - 1;
			ai_idx += ai_idx >> 5;
			bi_idx += bi_idx >> 5;

			temp[bi_idx] += temp[ai_idx];
		}
		offset <<= 1;
	}

	// clear last element and save block sum
	if(tid == 0) {
		int last_idx = num_elements - 1;
		last_idx += (last_idx >> 5);

		block_sums[gid] = temp[last_idx];
		temp[last_idx] = ZERO;
	}

	// down-sweep phase
	for(int d = 1; d < num_elements; d <<= 1) {
		offset >>= 1;
		barrier(CLK_LOCAL_MEM_FENCE);
		if(tid < d) {
			int ai_idx = offset * ((tid << 1) + 1) - 1;
			int bi_idx = offset * ((tid << 1) + 2) - 1;
			ai_idx += (ai_idx >> 5);
			bi_idx += (bi_idx >> 5);

			VALUE t = temp[ai_idx];
			temp[ai_idx] = temp[bi_idx];
			temp[bi_idx] += t;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write results to device memory
	if(g_ai < n) { g_odata[g_ai] = temp[ai + bank_offset_a] + g_idata[g_ai]; }
	if(g_bi < n) { g_odata[g_bi] = temp[bi + bank_offset_b] + g_idata[g_bi]; }
}

__kernel void add_block_sums(__global VALUE* g_data, __global VALUE* block_sums, int n) {
	int tid = get_local_id(0);
	int gid = get_group_id(0);
	int local_size = get_local_size(0);

	if(gid > 0) {
		VALUE sum = block_sums[gid - 1];

		int g_ai = gid * (local_size * 2) + tid;
		int g_bi = g_ai + local_size;

		if(g_ai < n) { g_data[g_ai] += sum; }
		if(g_bi < n) { g_data[g_bi] += sum; }
	}
}

#endif