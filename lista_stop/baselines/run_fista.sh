#!/bin/bash

export CUDA_VISIBLE_DEVICES=-1

mix='20-30-40'

## 20 iterations
python3 fista.py \
    -num_algo_itr 20\
    -L 4 \
    -rho 0.21 \
    -snr_mix $mix

# 100 iterations
python3 fista.py \
    -num_algo_itr 100\
    -L 4.5 \
    -rho 0.05 \
    -snr_mix $mix