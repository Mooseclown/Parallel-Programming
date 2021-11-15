#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <emmintrin.h>
#include <nmmintrin.h>

#define ROUND_DOWN(x, align) ((int)(x) & ~(align - 1))
#define CHUNK  100

typedef struct process_info {
    int rank;
    int tasks;
    int ncpus;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
} process_info;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

void copy_row_to_image(int *image, int *row, int row_idx, int width) {
    int base_index = row_idx * width;
    for (int i = 0; i < width; ++i) {
        image[base_index + i] = row[i];
    }
}

void mandelbrot_set(process_info *p_info, int y_idx, int *image, int ncpus) {
    /* init some variables that will be used */
    double x_interval = (p_info->right - p_info->left) / p_info->width;
    double y_interval = (p_info->upper - p_info->lower) / p_info->height;
    int round_down_width = ROUND_DOWN(p_info->width, 2);

    int image_base = y_idx * p_info->width;
    double y0 = y_idx * y_interval + p_info->lower;

    omp_set_num_threads(ncpus);
    #pragma omp parallel
    {
        /* Vectorization */
        __m128d x0_128d, y0_128d, x_128d, y_128d, temp_128d;
        __m128d x_interval_128d, left_128d;
        __m128d length_squared_128d, x_square_128d, y_square_128d;
        __m128d two, four, cmp_ge_four;
        __m128i repeats_128i, flag_128i, iters_128i, cmp_iters;

        x_interval_128d = _mm_set_pd1(x_interval);
        left_128d = _mm_set_pd1(p_info->left);
        two = _mm_set_pd1(2);
        four = _mm_set_pd1(4);
        iters_128i = _mm_set1_epi64x(p_info->iters - 1);
        y0_128d = _mm_set_pd1(y0);

        /* Vectorization */
        #pragma omp for schedule(dynamic, CHUNK)
        for (int x_idx = 0; x_idx < round_down_width; x_idx += 2) {
            x0_128d = _mm_set_pd(x_idx + 1, x_idx);
            x0_128d = _mm_mul_pd(x0_128d, x_interval_128d);
            x0_128d = _mm_add_pd(x0_128d, left_128d);
            
            x_128d = _mm_setzero_pd();
            y_128d = _mm_setzero_pd();
            length_squared_128d = _mm_setzero_pd();
            repeats_128i = _mm_setzero_si128();
            flag_128i = _mm_set1_epi64x(1);

            while (flag_128i[0] || flag_128i[1]) {
                /* double temp = x * x - y * y + x0 */
                x_square_128d = _mm_mul_pd(x_128d, x_128d);             // x * x
                y_square_128d = _mm_mul_pd(y_128d, y_128d);             // y * y
                temp_128d = _mm_sub_pd(x_square_128d, y_square_128d);   // x * x - y * y
                temp_128d = _mm_add_pd(temp_128d, x0_128d);             // x * x - y * y + x0

                /* y = 2 * x * y + y0 */
                y_128d = _mm_mul_pd(x_128d, y_128d);                    // x * y
                y_128d = _mm_mul_pd(y_128d, two);                       // 2 * x * y
                y_128d = _mm_add_pd(y_128d, y0_128d);                   // 2 * x * y + y0
                
                /* x = temp */
                x_128d = temp_128d;

                /* length_squared = x * x + y * y */
                x_square_128d = _mm_mul_pd(x_128d, x_128d);             // x * x
                y_square_128d = _mm_mul_pd(y_128d, y_128d);             // y * y
                length_squared_128d = _mm_add_pd(x_square_128d, y_square_128d);       // x * x + y * y

                /* ++repeats */
                repeats_128i = _mm_add_epi64(repeats_128i, flag_128i);

                /* repeats < p_info->iters */
                cmp_iters = _mm_cmpgt_epi64(repeats_128i, iters_128i);
                /* length_squared < 4 */
                cmp_ge_four = _mm_cmpge_pd(length_squared_128d, four);

                if (flag_128i[0] && cmp_ge_four[0] || cmp_iters[0]) {
                    flag_128i[0] = 0;
                }

                if (flag_128i[1] && cmp_ge_four[1] || cmp_iters[1]) {
                    flag_128i[1] = 0;
                }
            }
            
            image[image_base + x_idx] = repeats_128i[0];
            image[image_base + x_idx + 1] = repeats_128i[1];
        }
    }

    /* remaining data */
    for (int x_idx = round_down_width; x_idx < p_info->width; ++x_idx) {
        double x0 = x_idx * x_interval + p_info->left;
        
        int repeats = 1;
        double x = x0;
        double y = y0;
        double length_squared = x * x + y * y;
        while (repeats < p_info->iters && length_squared < 4) {
            double temp = x * x - y * y + x0;
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        image[image_base + x_idx] = repeats;
    }
}

void slave(process_info *p_info) {
    /* allocate memory for image */
    int* image = (int*)malloc(p_info->width * p_info->height * sizeof(int));
    assert(image);
    
    int row_idx = p_info->rank;
    MPI_Status recv_status;
    do {
        mandelbrot_set(p_info, row_idx, image, p_info->ncpus);

        /* asyn send row buffer */
        MPI_Send(&image[row_idx * p_info->width], p_info->width, MPI_INT, p_info->tasks - 1, row_idx, MPI_COMM_WORLD);


        MPI_Recv(&row_idx, 1, MPI_INT, p_info->tasks - 1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);

    } while (row_idx < p_info->height);

    free(image);
}

void master(process_info *p_info, const char *filename) {
    /* allocate memory for image */
    int* image = (int*)malloc(p_info->width * p_info->height * sizeof(int));
    assert(image);

    int next_row = p_info->tasks;
    omp_lock_t next_row_lock;
    omp_init_lock(&next_row_lock);

    int remote = p_info->tasks - 1;

    #pragma omp parallel num_threads(2) default(shared)
    {
        #pragma omp sections
        {
            /* receive row buffer and write png */
            #pragma omp section
            {
                /* allocate memory for row buffer */
                int *row_buffer = (int *)malloc(p_info->width * sizeof(int));

                int recv_tasks = 0, buffer_idx, temp;
                MPI_Request recv_request;
                MPI_Status recv_status;
                while (1) {
                    memset(row_buffer, 0, p_info->width * sizeof(int));
                    MPI_Recv(row_buffer, p_info->width, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
                    
                    omp_set_lock(&next_row_lock);
                    temp = next_row++;
                    omp_unset_lock(&next_row_lock);

                    MPI_Send(&temp, 1, MPI_INT, recv_status.MPI_SOURCE, 0, MPI_COMM_WORLD);

                    buffer_idx = recv_status.MPI_TAG;
                    copy_row_to_image(image, row_buffer, buffer_idx, p_info->width);
                    ++recv_tasks;
                    
                    if (temp >= p_info->height) {
                        if (--remote == 0) {
                            break;
                        }
                    }
                }
                free(row_buffer);
            }

            /* mandelbrot_set */
            #pragma omp section
            {
                int row_idx = p_info->rank;
                do {
                    mandelbrot_set(p_info, row_idx, image, p_info->ncpus - 1);

                    /* get next row index */
                    omp_set_lock(&next_row_lock);
                    row_idx = next_row++;
                    omp_unset_lock(&next_row_lock);
                } while (row_idx < p_info->height);
            }
        }
    }
    
    write_png(filename, p_info->iters, p_info->width, p_info->height, image);

    omp_destroy_lock(&next_row_lock);

    free(image);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long ncpus = CPU_COUNT(&cpu_set);

    /* MPI init */
    int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); // the total number of process
	MPI_Comm_size(MPI_COMM_WORLD, &size); // the rank (id) of the calling process

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    process_info p_info;
    p_info.rank = rank;
    p_info.tasks = size;
    p_info.ncpus = (int)ncpus;
    p_info.iters = iters;
    p_info.left = left;
    p_info.right = right;
    p_info.lower = lower;
    p_info.upper = upper;
    p_info.width = width;
    p_info.height = height;

    if (rank == size - 1) {
        master(&p_info, filename);
    } else {
        // compute node
        slave(&p_info);
    }

    MPI_Finalize();
    return 0;
}
