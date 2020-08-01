#!/bin/bash

export CUDA_VISIBLE_DEVICES=-1

mix='20-30-40'

# 20 iterations
python3 ista.py \
    -num_algo_itr 20 \
    -L 2.7 \
    -rho 0.2 \
    -snr_mix $mix

# 100 iterations
python3 ista.py \
    -num_algo_itr 100 \
    -L 3 \
    -rho 0.11 \
    -snr_mix $mix