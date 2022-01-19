#!/bin/bash

make
rm ./metadata/*
rm ./output/*

filename=03

nodes=$(jq .NODES ./testcases/${filename}.json)
cpus=$(jq .CPUS ./testcases/${filename}.json)
job_name=$(jq -r .JOB_NAME ./testcases/${filename}.json)
num_reducer=$(jq .NUM_REDUCER ./testcases/${filename}.json)
delay=$(jq .DELAY ./testcases/${filename}.json)
input_file_name=$(jq -r .INPUT_FILE_NAME ./testcases/${filename}.json)
chunk_size=$(jq .CHUNK_SIZE ./testcases/${filename}.json)
locality_config_filename=$(jq -r .LOCALITY_CONFIG_FILENAME ./testcases/${filename}.json)

srun -N${nodes} -c${cpus} hw4 ${job_name} ${num_reducer} ${delay} \
/home/pp21/share/hw4/testcases/${input_file_name} \
${chunk_size} /home/pp21/share/hw4/testcases/${locality_config_filename} $(pwd)/output

make clean