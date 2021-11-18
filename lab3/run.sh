#!/bin/bash

png_name=large-candy

srun -n 1 --gres=gpu:1 ./lab3 testcases/${png_name}.png output/${png_name}.png