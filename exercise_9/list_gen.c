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

	for(long i = 0; i < N; ++i) {
		person_t p;
		gen_name(p.name);
		p.age = rand() % (MAX_AGE + 1);
		printf("%3d | %s\n", p.age, p.name);
	}

	return EXIT_SUCCESS;
}
