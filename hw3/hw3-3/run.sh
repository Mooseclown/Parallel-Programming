#!/bin/bash

filename=c05.1
#c20.1
#p12k1

make

#srun -p prof -N1 -n1 -c2 --gres=gpu:1 nvprof --metrics gld_throughput ./hw3-3 ./cases/${filename} ./output/${filename}.out
srun -N1 -n1 -c2 --gres=gpu:2 nvprof ./hw3-3 ./cases/${filename} ./output/${filename}.out
#srun -N1 -n1 -c2 --gres=gpu:2 cuda-memcheck ./hw3-3 ./cases/${filename} ./output/${filename}.out

sha256sum ./output/${filename}.out | awk '{print $1}' > ./output/${filename}.out.sha

diff ./output/${filename}.out.sha ./cases/${filename}.out.sha256

rm -f ./output/${filename}.out.sha
make clean