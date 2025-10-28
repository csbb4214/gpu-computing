#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// allows the user to specify the problem size at compile time
#ifndef N
#define N 1024
#endif
#ifndef IT
#define IT 100
#endif
#ifdef FLOAT
#define VALUE float
#else
#define VALUE double
#endif

VALUE u[N][N], tmp[N][N], f[N][N];

VALUE init_func(int x, int y) {
	return 40 * sin((VALUE)(16 * (2 * x - 1) * y));
}

int main(void) {
	// init matrix
	memset(u, 0, sizeof(u));

	// init F
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			f[i][j] = init_func(i, j);
		}
	}

	const VALUE factor = pow((VALUE)1 / N, 2);

	const double start_time = omp_get_wtime();

	// main Jacobi loop
	for(int it = 0; it < IT; it++) {
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
		for(int i = 1; i < N - 1; i++) {
			for(int j = 1; j < N - 1; j++) {
				tmp[i][j] = (VALUE)1 / 4 * (u[i - 1][j] + u[i][j + 1] + u[i][j - 1] + u[i + 1][j] - factor * f[i][j]);
			}
		}
		memcpy(u, tmp, N * N * sizeof(VALUE));
	}

	const double elapsed_ms = (omp_get_wtime() - start_time) * 1000.0;

	#ifdef FLOAT
	const char* prec = "float";
	#else
	const char* prec = "double";
	#endif

	#ifdef _OPENMP
	printf("openmp,%s,%d,%d,%.3f\n", prec, N, IT, elapsed_ms);
	#else
	printf("serial,%s,%d,%d,%.3f\n", prec, N, IT, elapsed_ms);
	#endif

	return EXIT_SUCCESS;
}
