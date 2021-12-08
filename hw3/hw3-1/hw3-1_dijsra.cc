#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

#define unlikely(x)  __builtin_expect(!!(x), 0)

#define INVALID_D 1073741823

struct thread_info {
    int tid;
    int V;
    unsigned int *G;
};

static unsigned int *read_file(const char *filename, int &V, int &E) {
    FILE *f;
    f = fopen(filename, "rb");
    assert(f);

    fread(&V, sizeof(int), 1, f);
    fread(&E, sizeof(int), 1, f);
    
    unsigned int *G = (unsigned int *)malloc(V * V * sizeof(unsigned int));

    /* init shortest-path distance array*/
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (unlikely(i == j)) {
                G[i * V + j] = 0;
            } else {
                G[i * V + j] = INVALID_D;
            }
        }
    }

    for(int i = 0; i < E; ++i) {
        int src, dst;
        fread(&src, sizeof(int), 1, f);
        fread(&dst, sizeof(int), 1, f);
        fread(G + src * V + dst, sizeof(int), 1, f);
    }
    
    fclose(f);
    return G;
}

static void write_file(const char *filename, int V, unsigned int *G) {
    FILE *f;
    f = fopen(filename, "wb+");
    assert(f);

    fwrite(G, sizeof(int), V * V, f);

    fclose(f);
}

static int get_min_distance(int V, unsigned int *d, int *check) {
    int min = INVALID_D, index;
    for (int i = 0; i < V; ++i) {
        if (!check[i] && d[i] < min) {
            min = d[i];
            index = i;
        }
    }
    return index;
}

static void dijkstra(int V, unsigned int *G, int src, unsigned int *d) {
    int check[V];

    for (int i = 0; i < V; ++i) {
        check[i] = 0;
        d[i] = INVALID_D;
    }

    d[src] = 0;

    for (int i = 0; i < V; ++i) {
        int m = get_min_distance(V, d, check);
        check[m] = 1;

        for (int j = 0; j < V; ++j) {
            if (!check[j] && G[m * V + j] != INVALID_D && d[m] + G[m * V + j] < d[j]) {
                d[j] = d[m] + G[m * V + j];
            }
        }
    }
}

int main(int argc, char **argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long ncpus = CPU_COUNT(&cpu_set);

    assert(argc == 3);
    const char *input = argv[1];
    const char *output = argv[2];

    int V, E;
    unsigned int *G, *d;
    G = read_file(input, V, E);

    d = (unsigned int *)malloc(V * V * sizeof(unsigned int));
    
    omp_set_num_threads(ncpus);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < V; ++i) {
        printf("src = %d, thread = %d\n", i, omp_get_thread_num());
        dijkstra(V, G, i, d + i * V);
    }

    write_file(output, V, d);
    free(G);

    return 0;
}