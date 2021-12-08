#!/bin/bash

filename=c01.1

make

srun -N1 -n1 -c4 ./hw3-1 ./cases/${filename} ./output/${filename}.out

hw3-cat ./cases/${filename}.out > answer
hw3-cat ./output/${filename}.out > out

diff ./answer ./out

make clean