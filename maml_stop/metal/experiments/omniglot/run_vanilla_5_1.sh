#!/bin/bash

data_name=omniglot
data_root=../../../dataset
num_ways=5
num_train_shots=1
num_unroll=3
lr_inner=0.05
pos_dist=geo
vi_net=shared
entropy_penalty=0.001

save_dir=../../../results/metal/vi-${vi_net}-optstop-${data_name}-nw-${num_ways}-ns-${num_train_shots}-nr-${num_unroll}-pos-${pos_dist}-e-${entropy_penalty}
export CUDA_VISIBLE_DEVICES=0

python ../main_opt_stop.py \
    -data_root $data_root \
    -save_dir $save_dir \
    -vi_net $vi_net \
    -pos_dist $pos_dist \
    -data_name $data_name \
    -num_ways $num_ways \
    -num_unroll $num_unroll \
    -num_train_shots $num_train_shots \
    -batches_per_val 100 \
    -lr_inner $lr_inner \
    -entropy_penalty $entropy_penalty \
    $@

