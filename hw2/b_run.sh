#!/bin/bash

id=01

make all

srun -N ${nodes} -n ${procs} hw1 ${data_len} testcases/${id}.in output/${id}.out

# hw1-floats testcases/${id}.out > check_output/${id}.answer
# hw1-floats output/${id}.out > check_output/${id}.output

# diff check_output/${id}.answer check_output/${id}.output
