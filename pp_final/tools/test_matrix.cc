#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <algorithm>

int main(int argc, char **argv) {
        unsigned int *in, *out;
        int in_cols, in_rows, out_cols, out_rows;
        
        assert(argc==3);

        FILE *f_in, *f_out;
        f_in = fopen(argv[1], "rb");
        f_out = fopen(argv[2], "rb");
        assert(f_in);
        assert(f_out);

        fread(&in_cols, sizeof(unsigned int), 1, f_in);
        fread(&in_rows, sizeof(unsigned int), 1, f_in);

        fread(&out_cols, sizeof(unsigned int), 1, f_out);
        fread(&out_rows, sizeof(unsigned int), 1, f_out);

        assert(in_cols == out_rows);
        assert(in_rows == out_cols);

        in = (unsigned int *)malloc(in_cols * in_rows * sizeof(unsigned int));
        out = (unsigned int *)malloc(out_cols * out_rows * sizeof(unsigned int));

        fread(in, sizeof(unsigned int), in_cols * in_rows, f_in);
        fread(out, sizeof(unsigned int), out_cols * out_rows, f_out);

        fclose(f_in);
        fclose(f_out);

        for (int i = 0; i < in_rows; ++i) {
             for (int j = 0; j < in_cols; ++j) {
                     if (in[j * in_rows + i] != out[i * out_rows + j]) {
                             printf("Dame, it's wrong at (%d, %d)  (%d, %d).\n", i, j, in[j * in_rows + i], out[i * out_rows + j]);
                             return 1;
                     }
             }   
        }
        printf("It's fine.\n");
        return 0;
}