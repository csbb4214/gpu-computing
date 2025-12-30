#include "clu_errcheck.h"
#include "clu_setup.h"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "resources/people.h"

#define WORKGROUP_SIZE 256

int main(int argc, char* argv[]) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s N [seed]\n", argv[0]);
		return EXIT_FAILURE;
	}

	const int N = atoi(argv[1]);
	if(N <= 0) {
		fprintf(stderr, "N must be > 0\n");
		return EXIT_FAILURE;
	}

	unsigned seed = (argc >= 3) ? (unsigned)strtoul(argv[2], NULL, 10) : (unsigned)time(NULL);
	srand(seed);

	/* ---------- Generate/print unsorted list ---------- */
	person_t* people = malloc(N * sizeof(person_t));
	person_t* sorted = malloc(N * sizeof(person_t));
	int* ages = malloc(N * sizeof(int));

	for(int i = 0; i < N; ++i) {
		gen_name(people[i].name);
		people[i].age = rand() % (MAX_AGE + 1);
		ages[i] = people[i].age;
	}

	printf("Unsorted:\n");
	for(int i = 0; i < N; ++i) {
		printf("%3d | %s\n", people[i].age, people[i].name);
	}

	/* ---------- (1) Histogram ---------- */
	clu_env env;
	cl_queue_properties queue_props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};

	if(clu_initialize(&env, queue_props) != 0) {
		fprintf(stderr, "Failed to initialize OpenCL\n");
		return EXIT_FAILURE;
	}

	// Load kernel
	size_t source_size = 0;
	char* source_str = clu_load_kernel_source("histogram.cl", &source_size);
	if(!source_str) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_program program = clu_create_program(env.context, env.device_id, source_str, source_size, NULL);
	free(source_str);

	if(!program) {
		clu_release(&env);
		return EXIT_FAILURE;
	}

	cl_int err;
	cl_kernel kernel = clCreateKernel(program, "histogram", &err);
	CLU_ERRCHECK(err);

	// Buffers
	int histogram[MAX_AGE + 1] = {0};

	cl_mem buf_ages = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(int), ages, &err);
	CLU_ERRCHECK(err);

	cl_mem buf_hist = clCreateBuffer(env.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, (MAX_AGE + 1) * sizeof(int), histogram, &err);
	CLU_ERRCHECK(err);

	// Kernel arguments
	CLU_ERRCHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_ages));
	CLU_ERRCHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_hist));
	CLU_ERRCHECK(clSetKernelArg(kernel, 2, sizeof(int), &N));

	// Launch kernel
	size_t global_size = ((size_t)N + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
	size_t local_size = WORKGROUP_SIZE;

	cl_event event_kernel, event_read;

	CLU_ERRCHECK(clEnqueueNDRangeKernel(env.command_queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, &event_kernel));

	// Read back histogram
	CLU_ERRCHECK(clEnqueueReadBuffer(env.command_queue, buf_hist, CL_TRUE, 0, (MAX_AGE + 1) * sizeof(int), histogram, 0, NULL, &event_read));

	/* ---------- (2) Prefix-Sum ---------- */
	int sum = 0;
	for(int age = 0; age <= MAX_AGE; ++age) {
		int tmp = histogram[age];
		histogram[age] = sum;
		sum += tmp;
	}

	/* ---------- (3) Sorted Insertion ---------- */
	for(int i = 0; i < N; ++i) {
		int age = people[i].age;
		int pos = histogram[age];
		sorted[pos] = people[i];
		histogram[age]++;
	}

	printf("\nSorted:\n");
	for(int i = 0; i < N; ++i) {
		printf("%3d | %s\n", sorted[i].age, sorted[i].name);
	}

	/* ---------- Cleanup ---------- */
	clReleaseEvent(event_kernel);
	clReleaseEvent(event_read);
	clReleaseMemObject(buf_hist);
	clReleaseMemObject(buf_ages);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clu_release(&env);

	free(sorted);
	free(ages);
	free(people);

	return EXIT_SUCCESS;
}
