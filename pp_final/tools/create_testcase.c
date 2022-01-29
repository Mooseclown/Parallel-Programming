#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

int main(int argc, char **argv) {
    FILE *in, *out;
    int r;

    assert(argc == 5);
    
    int row_size = atoi(argv[1]);
    int col_size = atoi(argv[2]);
    assert(col_size <= 30000);
    assert(row_size <= 30000);

    const char *in_name = argv[3];
    const char *out_name = argv[4];

    in = fopen(in_name, "wb+");
    out = fopen(out_name, "wb+");

    fwrite(&row_size, sizeof(unsigned int), 1, in);
    fwrite(&col_size, sizeof(unsigned int), 1, in);

    fwrite(&col_size, sizeof(unsigned int), 1, out);
    fwrite(&row_size, sizeof(unsigned int), 1, out);

    unsigned int *matrix = (unsigned int *)malloc(col_size * row_size * sizeof(unsigned int));

    srand(time(NULL));

    for (int i = 0; i < col_size * row_size; ++i) {
        matrix[i] = rand();
    }

    for (int i = 0; i < row_size; ++i) {
        for (int j = 0; j < col_size; ++j) {
            r = fwrite(matrix + j * row_size + i, sizeof(unsigned int), 1, in);
            if (r < 0) {
                perror("write error.\n");
            }
        }
    }

    for (int i = 0; i < col_size; ++i) {
        for (int j = 0; j < row_size; ++j) {
            r = fwrite(matrix + i * row_size + j, sizeof(unsigned int), 1, out);
            if (r < 0) {
                perror("write error.\n");
            }
        }
    }

    fclose(in);
    fclose(out);
}