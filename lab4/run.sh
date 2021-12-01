#!/bin/bash

png_name=candy

make
srun -n 1 --gres=gpu:1 ./lab4 testcases/${png_name}.png output/${png_name}.png
#srun -n 1 --gres=gpu:1 cuda-memcheck ./lab4 testcases/${png_name}.png output/${png_name}.png
#srun -n 1 --gres=gpu:1 nvprof ./lab4 testcases/${png_name}.png output/${png_name}.png

png-diff testcases/${png_name}.out.png output/${png_name}.png
make clean
