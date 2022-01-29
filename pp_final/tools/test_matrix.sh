#!/bin/bash

files=$(ls testcases | grep txt)

for file in ${files[@]}
do 
    filename=$(echo $file | awk '{print substr($0,1,6)}')
    ./test_matrix ./testcases/${filename}.in ./testcases/${filename}.out
done
