#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define MAX_COMPONENTS 4

// Reduction kernel to find min, max, and sum for each component
// Each workgroup handles multiple pixels and reduces them locally
// Output format: for each workgroup and component -> [min, max, sum]
__kernel void reduce_stats(const __global uchar* image, __global ulong* stats, const int width, const int height, const int components) {
	const int total_pixels = width * height;
	const int gid = get_global_id(0);
	const int lid = get_local_id(0);
	const int workgroup_size = get_local_size(0);
	const int workgroup_id = get_group_id(0);

	// Local memory for reduction within workgroup - use uint for atomic operations (only type supported)
	__local uint local_min[MAX_COMPONENTS];
	__local uint local_max[MAX_COMPONENTS];
	__local uint local_sum[MAX_COMPONENTS];

	// Initialize local memory
	for(int c = 0; c < components; c++) {
		if(lid == 0) {
			local_min[c] = 255;
			local_max[c] = 0;
			local_sum[c] = 0;
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Process pixels assigned to this work item
	int pixels_per_item = (total_pixels + get_global_size(0) - 1) / get_global_size(0);
	int start_pixel = gid * pixels_per_item;
	int end_pixel = min(start_pixel + pixels_per_item, total_pixels);

	for(int pixel = start_pixel; pixel < end_pixel; pixel++) {
		int base_idx = pixel * components;

		for(int c = 0; c < components; c++) {
			uint val = (uint)image[base_idx + c];

			// Update local min
			atomic_min(&local_min[c], val);

			// Update local max
			atomic_max(&local_max[c], val);

			// Update local sum
			atomic_add(&local_sum[c], val);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Write results to global memory (workgroup 0, component c writes to position [c][0,1,2])
	if(lid == 0) {
		for(int c = 0; c < components; c++) {
			int base_idx = (workgroup_id * components + c) * 3;
			stats[base_idx] = (ulong)local_min[c];
			stats[base_idx + 1] = (ulong)local_max[c];
			stats[base_idx + 2] = (ulong)local_sum[c];
		}
	}
}

// Kernel to adjust image levels based on calculated factors
__kernel void adjust_levels(const __global uchar* input, __global uchar* output, const __global uchar* avg, const __global float* min_fac,
    const __global float* max_fac, const int width, const int height, const int components) {
	const int x = get_global_id(0);
	const int y = get_global_id(1);

	if(x >= width || y >= height) return;

	const int pixel_index = (y * width + x) * components;

	for(int c = 0; c < components; c++) {
		uchar val = input[pixel_index + c];
		uchar avg_val = avg[c];
		float v = (float)val - (float)avg_val;

		if(val < avg_val) {
			v *= min_fac[c];
		} else {
			v *= max_fac[c];
		}

		// Clamp to 0-255 range
		int result = (int)(v + avg_val + 0.5f); // +0.5 for rounding
		if(result < 0) result = 0;
		if(result > 255) result = 255;

		output[pixel_index + c] = (uchar)result;
	}
}