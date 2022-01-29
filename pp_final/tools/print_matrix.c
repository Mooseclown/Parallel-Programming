#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <assert.h>

int main(int argc, char **argv) {
    FILE *f;
    int rows, cols;

    assert(argc == 2);
    const char *file_name = argv[1];

    f = fopen(file_name, "rb");

    fread(&cols, sizeof(unsigned int), 1, f);
    fread(&rows, sizeof(unsigned int), 1, f);
    printf("cols: %d, rows: %d.\n", cols, rows);

    unsigned int *matrix = (unsigned int *)malloc(rows * cols * sizeof(unsigned int));

    fread(matrix, sizeof(unsigned int), rows * cols, f);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%u ", matrix[j * rows + i]);
        }
        printf("\n");
    }

    fclose(f);
}