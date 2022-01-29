#!/bin/bash

filename=c11.1

make

#srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput ./hw3-2 ./cases/${filename} ./output/${filename}.out
srun -N1 -n1 --gres=gpu:1 nvprof ./hw3-2 ./cases/${filename} ./output/${filename}.out
#srun -N1 -n1 --gres=gpu:1 cuda-memcheck ./hw3-2 ./cases/${filename} ./output/${filename}.out

sha256sum ./output/${filename}.out | awk '{print $1}' > ./output/${filename}.out.sha

diff ./output/${filename}.out.sha ./cases/${filename}.out.sha256

make clean