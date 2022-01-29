#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>

#define B 64
#define thread_num 32

#define unlikely(x)  __builtin_expect(!!(x), 0)
#define ROUND_UP(x, align) (((int) (x) + (align - 1)) & ~(align - 1))

const int INF = ((1 << 30) - 1);

int v_num, e_num, v_orig;

inline int ceil(int a, int b) { return (a + b - 1) / b; }

static unsigned int *input(const char *filename) {
    FILE *f;
    f = fopen(filename, "rb");
    assert(f);

    fread(&v_orig, sizeof(int), 1, f);
    fread(&e_num, sizeof(int), 1, f);
    v_num = ROUND_UP(v_orig, B);
    unsigned int *G;
    cudaMallocHost(&G, v_num * v_num * sizeof(unsigned int));

    /* init shortest-path distance array*/
    std::fill_n(G, v_num * v_num, INF);
    for (int i = 0; i < v_num; ++i) {
        G[i * v_num + i] = 0;
    }

    int *pairs = (int *)malloc(3 * e_num * sizeof(int));
    fread(pairs, sizeof(int), 3 * e_num, f);
    for (int i = 0; i < e_num; ++i) {
        int index = pairs[3 * i] * v_num + pairs[3 * i + 1];
        G[index] = pairs[3 * i + 2];
    }
    free(pairs);
    
    fclose(f);
    return G;
}

void output(char* outFileName, unsigned int *G) {
    FILE *f;
    f = fopen(outFileName, "wb+");
    assert(f);

    for (int i = 0; i < v_orig; ++i) {
        fwrite(G + i * v_num, sizeof(int), v_orig, f);
    }
    fclose(f);
}

__global__ void block_FW_phase1(int row_size, int round, unsigned int *dist) {
    __shared__ unsigned int s_dist[B][B];

    int block_start = round * B;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    /* pivot (global -> share) */
    #pragma unroll 2
    for (int i = 0; i < B; i += thread_num) {
        #pragma unroll 2
        for (int j = 0; j < B; j += thread_num) {
            int si = tid_y + i;
            int sj = tid_x + j;

            int global_idx = (block_start + si) * row_size + block_start + sj;

            s_dist[si][sj] = dist[global_idx];
        }
    }

    /* cal */
    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        #pragma unroll 2
        for (int i = 0; i < B; i += thread_num) {
            #pragma unroll 2
            for (int j = 0; j < B; j += thread_num) {
                int si = tid_y + i;
                int sj = tid_x + j;
                unsigned int tmp = s_dist[si][k] + s_dist[k][sj];
                s_dist[si][sj] = min(s_dist[si][sj], tmp);
            }
        }
    }

    /* pivot (share -> global) */
    #pragma unroll 2
    for (int i = 0; i < B; i += thread_num) {
        #pragma unroll 2
        for (int j = 0; j < B; j += thread_num) {
            int si = tid_y + i;
            int sj = tid_x + j;

            int global_idx = (block_start + si) * row_size + block_start + sj;

            dist[global_idx] = s_dist[si][sj];
        }
    }
}

__global__ void block_FW_phase2_row(int row_size, int round, unsigned int *dist) {    
    __shared__ unsigned int self[B][B];
    __shared__ unsigned int pivot[B][B];

    int pivot_start = round * B;
    int block_start = blockIdx.x * B;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    /* global -> share */
    #pragma unroll 2
    for (int x = 0; x < B; x += thread_num) {
        #pragma unroll 2
        for (int y = 0; y < B; y += thread_num) {
            int sx = tid_y + x;
            int sy = tid_x + y;

            int self_idx =  (block_start + sx) * row_size + pivot_start + sy;
            int pivot_idx = (pivot_start + sx) * row_size + pivot_start + sy;

            self[sx][sy] = dist[self_idx];
            pivot[sx][sy] = dist[pivot_idx];
        }
    }

    /* cal */
    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        #pragma unroll 2
        for (int i = 0; i < B; i += thread_num) {
            #pragma unroll 2
            for (int j = 0; j < B; j += thread_num) {
                int si = tid_y + i;
                int sj = tid_x + j;
                unsigned int tmp = self[si][k] + pivot[k][sj];
                self[si][sj] = min(self[si][sj], tmp);
            }
        }
    }
    
    /* self(i,k) (share -> global) */
    #pragma unroll 2
    for (int i = 0; i < B; i += thread_num) {
        #pragma unroll 2
        for (int k = 0; k < B; k += thread_num) {
            int si = tid_y + i;
            int sk = tid_x + k;

            int global_idx = (block_start + si) * row_size + pivot_start + sk;

            dist[global_idx] = self[si][sk];
        }
    }
}

__global__ void block_FW_phase2_column(int row_size, int round, unsigned int *dist) {
    __shared__ unsigned int self[B][B];
    __shared__ unsigned int pivot[B][B];

    int pivot_start = round * B;
    int block_start = blockIdx.x * B;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    /* global -> share */
    #pragma unroll 2
    for (int x = 0; x < B; x += thread_num) {
        #pragma unroll 2
        for (int y = 0; y < B; y += thread_num) {
            int sx = tid_y + x;
            int sy = tid_x + y;

            int self_idx = (pivot_start + sx) * row_size + block_start + sy;
            int pivot_idx = (pivot_start + sx) * row_size + pivot_start + sy;

            self[sx][sy] = dist[self_idx];
            pivot[sx][sy] = dist[pivot_idx];
        }
    }

    /* cal */
    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        __syncthreads();
        #pragma unroll 2
        for (int i = 0; i < B; i += thread_num) {
            #pragma unroll 2
            for (int j = 0; j < B; j += thread_num) {
                int si = tid_y + i;
                int sj = tid_x + j;
                unsigned int tmp = pivot[si][k] + self[k][sj];
                self[si][sj] = min(self[si][sj], tmp);
            }
        }
    }
    
    /* self(k,j) (share -> global) */
    #pragma unroll 2
    for (int k = 0; k < B; k += thread_num) {
        #pragma unroll 2
        for (int j = 0; j < B; j += thread_num) {
            int sk = tid_y + k;
            int sj = tid_x + j;

            int global_idx = (pivot_start + sk) * row_size + block_start + sj;
            
            dist[global_idx] = self[sk][sj];
        }
    }
}

__global__ void block_FW_phase3(int row_size, int round, int block_base_x, int block_base_y, unsigned int *dist) {
    __shared__ unsigned int ij[B][B];
    __shared__ unsigned int ik[B][B];
    __shared__ unsigned int kj[B][B];
    
    int block_start_j = (block_base_x + blockIdx.x) * B;
    int block_start_i = (block_base_y + blockIdx.y) * B;
    int block_start_k = round * B;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    /* global to share */
    #pragma unroll 2
    for (int x = 0; x < B; x += thread_num) {
        #pragma unroll 2
        for (int y = 0; y < B; y += thread_num) {
            int sx = tid_y + x;
            int sy = tid_x + y;

            int g_ij_idx = (block_start_i + sx) * row_size + block_start_j + sy;
            int g_ik_idx = (block_start_i + sx) * row_size + block_start_k + sy;
            int g_kj_idx = (block_start_k + sx) * row_size + block_start_j + sy;

            ij[sx][sy] = dist[g_ij_idx];
            ik[sx][sy] = dist[g_ik_idx];
            kj[sx][sy] = dist[g_kj_idx];
        }
    }

    __syncthreads();
    /* cal */
    #pragma unroll 64
    for (int k = 0; k < B; ++k) {
        #pragma unroll 2
        for (int i = 0; i < B; i += thread_num) {
            #pragma unroll 2
            for (int j = 0; j < B; j += thread_num) {
                int si = tid_y + i;
                int sj = tid_x + j;
                unsigned int tmp = ik[si][k] + kj[k][sj];
                ij[si][sj] = min(ij[si][sj], tmp);
            }
        }
    }

    /* i, j (share -> global) */
    #pragma unroll 2
    for (int i = 0; i < B; i += thread_num) {
        #pragma unroll 2
        for (int j = 0; j < B; j += thread_num) {
            int si = tid_y + i;
            int sj = tid_x + j;
            
            int global_idx = (block_start_i + si) * row_size + block_start_j + sj;
            
            dist[global_idx] = ij[si][sj];
        }
    }
}

int main(int argc, char* argv[]) {
    unsigned int *G, *d_dist;
    size_t pitch;
    G = input(argv[1]);

    cudaMallocPitch((void **)&d_dist, &pitch, v_num * sizeof(unsigned int), v_num);

    cudaMemcpy2D(d_dist, pitch, G, v_num * sizeof(unsigned int), v_num * sizeof(unsigned int), v_num, cudaMemcpyHostToDevice);

    int pitch_row_size = pitch / sizeof(unsigned int);
    int round = ceil(v_num, B);
    dim3 num_threads(thread_num, thread_num); // because num of threads per block is 1024
    dim3 p2_num_blocks(round, 1);
    dim3 p3_num_blocks;
    for (int r = 0; r < round - 1; ++r) {
        int tmp = round - (r + 1);
        
        block_FW_phase1 <<<1, num_threads>>> (pitch_row_size, r, d_dist);

        block_FW_phase2_row <<<p2_num_blocks, num_threads>>> (pitch_row_size, r, d_dist);

        block_FW_phase2_column <<<p2_num_blocks, num_threads>>> (pitch_row_size, r, d_dist);

        // /* Phase 3 */
        // if (r > 0) {
        //     /* 左上 */
        //     p3_num_blocks.x = r;
        //     p3_num_blocks.y = 1;
        //     for (int i = 0; i < r; ++i) {
        //         block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, 0, i, d_dist);
        //     }
        // }
        // if (tmp > 0) {
        //     /* 右下 */
        //     p3_num_blocks.x = tmp;
        //     p3_num_blocks.y = 1;
        //     for (int i = 0; i < tmp; ++i) {
        //         block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, r+1, r+1+i, d_dist);
        //     }
        // }
        // if (r > 0 && tmp > 0) {
        //     /* 左下 */
        //     p3_num_blocks.x = r;
        //     p3_num_blocks.y = 1;
        //     for (int i = 0; i < tmp; ++i) {
        //         block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, 0, r+1+i, d_dist);
        //     }
        //     /* 右上 */
        //     p3_num_blocks.x = tmp;
        //     p3_num_blocks.y = 1;
        //     for (int i = 0; i < r; ++i) {
        //         block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, r+1, i, d_dist);
        //     }
        // }

        /* Phase 3 */
        if (r > 0) {
            /* 左上 */
            p3_num_blocks.x = r;
            p3_num_blocks.y = r;
            block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, 0, 0, d_dist);
        }
        if (tmp > 0) {
            /* 右下 */
            p3_num_blocks.x = tmp;
            p3_num_blocks.y = tmp;
            block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, r+1, r+1, d_dist);
        }
        if (r > 0 && tmp > 0) {
            /* 左下 */
            p3_num_blocks.x = r;
            p3_num_blocks.y = tmp;
            block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, 0, r+1, d_dist);
            /* 右上 */
            p3_num_blocks.x = tmp;
            p3_num_blocks.y = r;
            block_FW_phase3 <<<p3_num_blocks, num_threads>>> (pitch_row_size, r, r+1, 0, d_dist);
        }

    }

    /* last round */
    block_FW_phase1 <<<1, num_threads>>> (pitch_row_size, round -1, d_dist);

    block_FW_phase2_row <<<p2_num_blocks, num_threads>>> (pitch_row_size, round -1, d_dist);

    block_FW_phase2_column <<<p2_num_blocks, num_threads>>> (pitch_row_size, round -1, d_dist);

    /* Phase 3 */
    /* 左上 */
    cudaStream_t streams[20];
    for (int i = 0; i < 20; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    p3_num_blocks.x = round - 1;
    p3_num_blocks.y = 1;
    for (int i = 0; i < round - 1; ++i) {
        block_FW_phase3 <<<p3_num_blocks, num_threads, 0, streams[i%20]>>> (pitch_row_size, round -1, 0, i, d_dist);
        
        cudaMemcpy2DAsync(G + i * B * v_num, v_num * sizeof(unsigned int),
                     d_dist + i * B * pitch_row_size, pitch,
                     v_num * sizeof(unsigned int), B, cudaMemcpyDeviceToHost, streams[i%20]);
    }

    for (int i = 0; i < 20; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    cudaMemcpy2D(G + (round - 1) * B * v_num, v_num * sizeof(unsigned int),
                 d_dist + (round - 1) * B * pitch_row_size, pitch,
                 v_num * sizeof(unsigned int), B, cudaMemcpyDeviceToHost);

    //cudaMemcpy2D(G, v_num * sizeof(unsigned int), d_dist, pitch, v_num * sizeof(unsigned int), v_num, cudaMemcpyDeviceToHost);

    cudaFree(d_dist);

    output(argv[2], G);
    cudaFreeHost(G);

    return 0;
}