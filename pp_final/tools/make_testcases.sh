#!/bin/bash

#files=$(ls testcases | grep txt)
files=blk_33.txt

for file in ${files[@]}
do 
    filename=$(echo $file | awk '{print substr($0,1,6)}')
    info=$(cat ./testcases/${filename}.txt)
    row_size=${info[0]}
    col_size=${info[1]}
    ./create_testcase ${row_size} ${col_size} ./testcases/${filename}.in ./testcases/${filename}.out
    sha256sum ./testcases/${filename}.out | awk '{print $1}' > ./testcases/${filename}.out.sha
done
