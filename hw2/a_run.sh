#!/bin/bash

make all

type=fast
id=01

info=$(cat ./testcases/${type}${id}.txt)
iter=${info[0]}
x0=${info[1]}
x1=${info[2]}
y0=${info[3]}
y1=${info[4]}
w=${info[5]}
h=${info[6]}

echo ${info}

srun -n1 -c4 ./hw2a ./output/${type}${id}.png ${iter} ${x0} ${x1} ${y0} ${y1} ${w} ${h}