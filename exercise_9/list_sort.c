#include "resources/people.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
	if(argc < 2) {
		fprintf(stderr, "Usage: %s N [seed]\n", argv[0]);
		return EXIT_FAILURE;
	}

	long N = strtol(argv[1], NULL, 10);
	if(N <= 0) {
		fprintf(stderr, "N must be > 0\n");
		return EXIT_FAILURE;
	}

	unsigned seed = (argc >= 3) ? (unsigned)strtoul(argv[2], NULL, 10) : (unsigned)time(NULL);
	srand(seed);

	/* ---------- Generate/print unsorted list ---------- */
	person_t* A = malloc(N * sizeof(person_t));
	person_t* B = malloc(N * sizeof(person_t));
	if(!A || !B) {
		perror("malloc");
		return EXIT_FAILURE;
	}

	for(long i = 0; i < N; ++i) {
		gen_name(A[i].name);
		A[i].age = rand() % (MAX_AGE + 1);
	}

	printf("Unsorted:\n");
	for(long i = 0; i < N; ++i) {
		printf("%3d | %s\n", A[i].age, A[i].name);
	}

	/* ---------- (1) Histogram ---------- */
	int C[MAX_AGE + 1] = {0};
	for(long i = 0; i < N; ++i) {
		C[A[i].age]++;
	}

	/* ---------- (2) Prefix-Sum ---------- */
	int sum = 0;
	for(int age = 0; age <= MAX_AGE; ++age) {
		int tmp = C[age];
		C[age] = sum;
		sum += tmp;
	}

	/* ---------- (3) Sorted Insertion ---------- */
	for(long i = 0; i < N; ++i) {
		int age = A[i].age;
		int pos = C[age];
		B[pos] = A[i];
		C[age]++;
	}

	printf("\nSorted:\n");
	for(long i = 0; i < N; ++i) {
		printf("%3d | %s\n", B[i].age, B[i].name);
	}

	free(A);
	free(B);
	return EXIT_SUCCESS;
}
