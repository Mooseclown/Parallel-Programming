#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sched.h>
#include <algorithm>

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height, int ncpus);

int n, m;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    int B = 512;
    block_FW(B);
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void block_FW(int B) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long ncpus = CPU_COUNT(&cpu_set);
    
    int round = ceil(n, B);
    for (int r = 0; r < round; ++r) {
        /* Phase 1*/
        cal(B, r, r, r, 1, 1, ncpus);

        /* Phase 2*/
        cal(B, r, r, 0, 1, r, ncpus);
        cal(B, r, r, r + 1, 1, round - r - 1, ncpus);
        cal(B, r, 0, r, r, 1, ncpus);
        cal(B, r, r + 1, r, round - r - 1, 1, ncpus);

        /* Phase 3*/
        cal(B, r, 0, 0, r, r, ncpus);
        cal(B, r, 0, r + 1, r, round - r - 1, ncpus);
        cal(B, r, r + 1, 0, round - r - 1, r, ncpus);
        cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1, ncpus);
    }
}

void cal(int B, int Round, int block_start_x, int block_start_y,
         int block_width, int block_height, int ncpus) {
    
    int block_end_x = block_start_x + block_width;
    int block_end_y = block_start_y + block_height;

    omp_set_num_threads(ncpus);
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                #pragma omp parallel for schedule(dynamic) default(shared)
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
                            Dist[i][j] = Dist[i][k] + Dist[k][j];
                        }
                    }
                }
            }
        }
    }
}