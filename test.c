#include <stdio.h>
#include <CL/cl.h>

int main() {
    cl_uint platformCount = 0;
    clGetPlatformIDs(0, NULL, &platformCount);
    printf("Found %d OpenCL platform(s)\n", platformCount);
    
    if (platformCount > 0) {
        cl_platform_id platforms[4];
        clGetPlatformIDs(platformCount, platforms, NULL);
        
        for (cl_uint i = 0; i < platformCount; i++) {
            char vendor[128];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
            printf("Platform %d: %s\n", i, vendor);
        }
    }
    
    return 0;
}