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
#include <pthread.h>
#include <emmintrin.h>
#include <nmmintrin.h>

#define ROUND_DOWN(x, align) ((int)(x) & ~(align - 1))

typedef struct thread_info {
    int tid;
    int ncpus;
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    void *private_data;
} thread_info;

typedef struct write_png_data {
    png_structp png_ptr;
    png_bytepp rows;
    size_t row_size;
    int *rows_bitmap;                // this bitmap record already colored row
    const int* buffer;
    int color_idx;                  // the last index of rows already allocate to color
    int png_idx;                    // the next index to be write png
    pthread_mutex_t color_idx_mutex;
    pthread_mutex_t png_mutex;
} write_png_data;

int next_color_index(write_png_data *wp_data, int max_index) {
    if (wp_data->color_idx < max_index) {
        return wp_data->color_idx++; 
    }
    return -1;
}

void rendring_row(write_png_data *wp_data, int idx, int max_idx) {
    if (pthread_mutex_trylock(&wp_data->png_mutex) == 0) {
        if (idx >= wp_data->png_idx) {
            while (wp_data->png_idx < max_idx && wp_data->rows_bitmap[wp_data->png_idx] == 1) {
                png_write_row(wp_data->png_ptr, wp_data->rows[wp_data->png_idx]);
                free(wp_data->rows[wp_data->png_idx++]);
            }
        }
        pthread_mutex_unlock(&wp_data->png_mutex);
    }
}

void *rendering(void *info) {
    thread_info *t_info = (thread_info *)info;
    write_png_data *wp_data = (write_png_data *)t_info->private_data;

    if (t_info->tid >= t_info->height) {
        pthread_exit(NULL);
    }

    /* set cpu core */
    int cpu_id = t_info->tid;
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    png_bytep row;
    int idx = t_info->tid;
    do {
        row = (png_bytep)malloc(wp_data->row_size);
        wp_data->rows[idx] = row;
        memset(wp_data->rows[idx], 0, wp_data->row_size);
        int buffer_base = (t_info->height - 1 - idx) * t_info->width;
        for (int x = 0; x < t_info->width; ++x) {
            int p = wp_data->buffer[buffer_base + x];
            png_bytep color = wp_data->rows[idx] + x * 3;
            if (p != t_info->iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        wp_data->rows_bitmap[idx] = 1;
        rendring_row(wp_data, idx, t_info->height);

        pthread_mutex_lock(&wp_data->color_idx_mutex);
        idx = next_color_index(wp_data, t_info->height);
        pthread_mutex_unlock(&wp_data->color_idx_mutex);
    } while (idx >= 0);

    pthread_exit(NULL);
}

void write_png(const char* filename, int iters, int width, int height, const int* buffer, unsigned long ncpus) {
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

    /* init rows*/
    png_bytepp rows = (png_bytepp)malloc(height * sizeof(png_bytep));
    size_t row_size = 3 * width * sizeof(png_byte);

    /* init rows bitmap */
    int *rows_bitmap = (int *)malloc(sizeof(int) * height);
    for (int i = 0; i < height; ++i) {
        rows_bitmap[i] = 0;
    }

    /* init wirte_png_data */
    struct write_png_data wp_data;
    wp_data.png_ptr = png_ptr;
    wp_data.rows = rows;
    wp_data.row_size = row_size;
    wp_data.rows_bitmap = rows_bitmap;
    wp_data.buffer = buffer;
    wp_data.color_idx = ncpus;
    wp_data.png_idx = 0;
    pthread_mutex_init(&wp_data.color_idx_mutex, 0);
    pthread_mutex_init(&wp_data.png_mutex, 0);

    /* creat thread */
    pthread_t threads[ncpus];
	thread_info t_info[ncpus];
    for (int i = 0; i < ncpus; ++i) {
        t_info[i].tid = i;
        t_info[i].ncpus = ncpus;
        t_info[i].iters = iters;
        t_info[i].width = width;
        t_info[i].height = height;
        t_info[i].private_data = (void *)&wp_data;

        int ret = pthread_create(&threads[i], NULL, rendering, (void *)&t_info[i]);
		if (ret) {
			printf("ERROR; return code from pthread_create() is %d\n", ret);
			exit(-1);
		}
    }

    for (int i = 0; i < ncpus; ++i) {
        pthread_join(threads[i], NULL);
    }

    pthread_mutex_destroy(&wp_data.color_idx_mutex);
    pthread_mutex_destroy(&wp_data.png_mutex);

    free(rows);
    free(rows_bitmap);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

typedef struct mandelbrot_set_data {
    pthread_mutex_t mutex;
    unsigned long next_y_idx;       // next y will be get
    int *image;
    double x_interval;
    double y_interval;
} mandelbrot_set_data;

int get_next_y_idx(mandelbrot_set_data *m_data, unsigned long max_idx) {
    if (m_data->next_y_idx < max_idx) {
        return m_data->next_y_idx++;
    }
    return -1;
}

void *mandelbrot_set(void *info) {
    thread_info *t_info = (thread_info *)info;
    mandelbrot_set_data *m_data = (mandelbrot_set_data *)t_info->private_data;

    /* set cpu core */
    int cpu_id = t_info->tid;
	cpu_set_t cpu_set;
	CPU_ZERO(&cpu_set);
	CPU_SET(cpu_id, &cpu_set);
	pthread_setaffinity_np(pthread_self(), sizeof(cpu_set), &cpu_set);

    /* Vectorization */
    __m128d y_idx_128d, x_idx_128d, x0_128d, y0_128d, x_128d, y_128d, temp_128d, a, b;
    __m128d x_interval_128d, y_interval_128d, lower_128d, left_128d;
    __m128d length_squared_128d, x_square_128d, y_square_128d;
    __m128d two, four, cmp_ge_four;
    __m128i repeats_128i, flag_128i, iters_128i, cmp_iters;

    //x_interval_128d = _mm_set_pd1(m_data->x_interval * 2);
    x_interval_128d = _mm_set_pd1(m_data->x_interval);
    left_128d = _mm_set_pd1(t_info->left);
    two = _mm_set_pd1(2);
    four = _mm_set_pd1(4);
    iters_128i = _mm_set1_epi64x(t_info->iters - 1);

    
    int y_idx = t_info->tid;
    int round_down_width = ROUND_DOWN(t_info->width, 2);
    do {
        int image_base = y_idx * t_info->width;
        double y0 = y_idx * m_data->y_interval + t_info->lower;
        y0_128d = _mm_set_pd1(y0);

        /* Vectorization */
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

                /* repeats < t_info->iters */
                cmp_iters = _mm_cmpgt_epi64(repeats_128i, iters_128i);
                /* length_squared < 4 */
                cmp_ge_four = _mm_cmpge_pd(length_squared_128d, four);

                if (cmp_ge_four[0] || cmp_iters[0]) {
                    flag_128i[0] = 0;
                }

                if (cmp_ge_four[1] || cmp_iters[1]) {
                    flag_128i[1] = 0;
                }
            }
            
            m_data->image[image_base] = repeats_128i[0];
            m_data->image[image_base + 1] = repeats_128i[1];
            image_base += 2;
        }

        /* remaining data */
        for (int x_idx = round_down_width; x_idx < t_info->width; ++x_idx) {
            double x0 = x_idx * m_data->x_interval + t_info->left;
            
            int repeats = 1;
            double x = x0;
            double y = y0;
            double length_squared = x * x + y * y;
            while (repeats < t_info->iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            m_data->image[image_base++] = repeats;
        }

        pthread_mutex_lock(&m_data->mutex);
        y_idx = get_next_y_idx(m_data, t_info->height);
        pthread_mutex_unlock(&m_data->mutex);
    } while (y_idx >= 0);

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    unsigned long ncpus = CPU_COUNT(&cpu_set);

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

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* init mandelbrot_set_data */
    struct mandelbrot_set_data m_data;
    m_data.next_y_idx = ncpus;
    m_data.image = image;
    m_data.x_interval = (right - left) / width;
    m_data.y_interval = (upper - lower) / height;
    pthread_mutex_init(&m_data.mutex, 0);

    /* create thread */
    pthread_t threads[ncpus];
	thread_info t_info[ncpus];
    for (int i = 0; i < ncpus; ++i) {
        int ret;
        t_info[i].tid = i;
        t_info[i].ncpus = ncpus;
        t_info[i].iters = iters;
        t_info[i].left = left;
        t_info[i].right = right;
        t_info[i].lower = lower;
        t_info[i].upper = upper;
        t_info[i].width = width;
        t_info[i].height = height;
        t_info[i].private_data = (void *)&m_data;

        ret = pthread_create(&threads[i], NULL, mandelbrot_set, (void *)&t_info[i]);
		if (ret) {
			printf("ERROR; return code from pthread_create() is %d\n", ret);
			exit(-1);
		}
    }

    for (int i = 0; i < ncpus; ++i) {
		pthread_join(threads[i], NULL);
	}

    pthread_mutex_destroy(&m_data.mutex);
    
    /* draw and cleanup */
    write_png(filename, iters, width, height, image, ncpus);
    free(image);
}
