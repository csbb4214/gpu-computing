#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "resources/people.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s N [seed]\n", argv[0]);
        return EXIT_FAILURE;
    }

    long N = strtol(argv[1], NULL, 10);
    if (N <= 0) {
        fprintf(stderr, "N must be > 0\n");
        return EXIT_FAILURE;
    }

    unsigned seed = (argc >= 3)
        ? (unsigned)strtoul(argv[2], NULL, 10)
        : (unsigned)time(NULL);
    srand(seed);

    /* 1) Liste erzeugen */
    person_t *list = malloc(N * sizeof(person_t));
    if (!list) {
        perror("malloc");
        return EXIT_FAILURE;
    }

    for (long i = 0; i < N; ++i) {
        gen_name(list[i].name);
        list[i].age = rand() % (MAX_AGE + 1);
    }

    /* Unsortierte Ausgabe */
    printf("Unsorted:\n");
    for (long i = 0; i < N; ++i) {
        printf("%3d | %s\n", list[i].age, list[i].name);
    }

    /* 2) Countsort */
    int count[MAX_AGE + 1] = {0};

    for (long i = 0; i < N; ++i) {
        count[list[i].age]++;
    }

    person_t *sorted = malloc(N * sizeof(person_t));
    if (!sorted) {
        perror("malloc");
        free(list);
        return EXIT_FAILURE;
    }

    int index = 0;
    for (int age = 0; age <= MAX_AGE; ++age) {
        for (long i = 0; i < N; ++i) {
            if (list[i].age == age) {
                sorted[index++] = list[i];
            }
        }
    }

    /* Sortierte Ausgabe */
    printf("\nSorted:\n");
    for (long i = 0; i < N; ++i) {
        printf("%3d | %s\n", sorted[i].age, sorted[i].name);
    }

    free(list);
    free(sorted);
    return EXIT_SUCCESS;
}
