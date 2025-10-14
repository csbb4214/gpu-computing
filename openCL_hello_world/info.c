#include <stdio.h>
#include <stdlib.h>

#include "clu_errcheck.h"

static void print_platform_string(cl_platform_id platform, cl_platform_info param, const char* label) {
	char buffer[10240] = {0};
	CLU_ERRCHECK_MSG(clGetPlatformInfo(platform, param, sizeof(buffer), buffer, NULL), "clGetPlatformInfo %s", label);
	printf("  %s = %s\n", label, buffer);
}

int main(void) {
	cl_uint platform_count = 0;
	CLU_ERRCHECK_MSG(clGetPlatformIDs(0, NULL, &platform_count), "clGetPlatformIDs (count)");
	printf("Number of platforms: %u\n", platform_count);

	if(platform_count == 0U) {
		puts("No OpenCL platforms found.");
		return EXIT_SUCCESS;
	}

	cl_platform_id* platforms = (cl_platform_id*)malloc(platform_count * sizeof(*platforms));
	if(platforms == NULL) {
		fprintf(stderr, "Failed to allocate memory for %u platforms.\n", platform_count);
		return EXIT_FAILURE;
	}

	CLU_ERRCHECK_MSG(clGetPlatformIDs(platform_count, platforms, NULL), "clGetPlatformIDs (list)");

	for(cl_uint i = 0; i < platform_count; ++i) {
		printf("\nPlatform %u:\n", i);

		cl_uint device_count = 0;
		CLU_ERRCHECK_MSG(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &device_count), "clGetDeviceIDs (count)");
		printf("  Total devices = %u\n", device_count);

		print_platform_string(platforms[i], CL_PLATFORM_PROFILE, "PROFILE");
		print_platform_string(platforms[i], CL_PLATFORM_VERSION, "VERSION");
		print_platform_string(platforms[i], CL_PLATFORM_NAME, "NAME");
		print_platform_string(platforms[i], CL_PLATFORM_VENDOR, "VENDOR");
		print_platform_string(platforms[i], CL_PLATFORM_EXTENSIONS, "EXTENSIONS");
	}

	free(platforms);
	return EXIT_SUCCESS;
}
