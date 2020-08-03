#!/bin/bash

data_name=omniglot
data_root=../../../dataset
num_ways=20
min_train=1
max_train=5
num_unroll=10
lr_inner=0.1
pos_dist=geo
vi_net=shared
entropy_penalty=0.01

save_dir=../../../results/metal-imba/vi-${vi_net}-optstop-${data_name}-nw-${num_ways}-ns-${min_train}-${max_train}-nr-${num_unroll}-pos-${pos_dist}-e-${entropy_penalty}-lr-${lr_inner}
export CUDA_VISIBLE_DEVICES=0

python ../main_opt_stop.py \
    -data_root $data_root \
    -save_dir $save_dir \
    -pos_dist $pos_dist \
    -vi_net $vi_net \
    -data_name $data_name \
    -num_ways $num_ways \
    -num_unroll $num_unroll \
    -min_train_shots $min_train \
    -max_train_shots $max_train \
    -batches_per_val 100 \
    -meta_batch_size 10 \
    -lr_inner $lr_inner \
    -entropy_penalty $entropy_penalty \
    -run_eval False \
    -sto_em True \
    -sto_maml True \
    $@

