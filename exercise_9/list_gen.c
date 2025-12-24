#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "resources/people.h"

int main(int argc, char* args[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s N [seed] [out.csv]\n", args[0]);
        return EXIT_FAILURE;
    }

    long N = strtol(args[1], NULL, 10);
    if(N <= 0) {
        fprintf(stderr, "N must be > 0\n");
        return EXIT_FAILURE;
    }

    unsigned seed = (argc >= 3) ? (unsigned)strtoul(args[2], NULL, 10) : (unsigned)time(NULL);
    srand(seed);

    person_t p;
    for(long i = 0; i < N; ++i) {
        gen_name(p.name);
        p.age = rand() % (MAX_AGE + 1);
        printf("\"%s\",%d\n", p.name, p.age);
    }

    return EXIT_SUCCESS;
}