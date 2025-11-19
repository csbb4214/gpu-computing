#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N
#define N 1024
#endif

#ifdef FLOAT
typedef float VALUE;
#else
typedef int VALUE;
#endif

VALUE arr[N];

int main(void) {
#ifdef FLOAT
	for(size_t i = 0; i < N; i++) {
		arr[i] = 1.0f;
	}
#else
	for(size_t i = 0; i < N; i++) {
		arr[i] = 1;
	}
#endif

	double t1 = omp_get_wtime();
	VALUE sum = 0;
	for(size_t i = 0; i < N; i++) {
		sum += arr[i];
	}
	double t2 = omp_get_wtime();
	sum == N ? printf("correct,%.3f\n", (t2 - t1)) : printf("wrong\n");

	return 0;
}
