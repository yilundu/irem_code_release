#!/bin/bash

data_name=omniglot
data_root=../../../dataset
num_ways=20
num_train_shots=1
num_unroll=5
lr_inner=0.1
pos_dist=geo
vi_net=shared
num_test=1
entropy_penalty=0.1
n_e=600

save_dir=../../../results/metal/vi-${vi_net}-optstop-${data_name}-nw-${num_ways}-ns-${num_train_shots}-nr-${num_unroll}-pos-${pos_dist}-e-${entropy_penalty}-nt-${num_test}

export CUDA_VISIBLE_DEVICES=0

python ../main_opt_stop.py \
    -data_root $data_root \
    -save_dir $save_dir \
    -pos_dist $pos_dist \
    -vi_net $vi_net \
    -data_name $data_name \
    -num_ways $num_ways \
    -num_unroll $num_unroll \
    -num_train_shots $num_train_shots \
    -batches_per_val 100 \
    -meta_batch_size 10 \
    -num_test_shots $num_test \
    -num_epochs $n_e \
    -lr_inner $lr_inner \
    -entropy_penalty $entropy_penalty \
    -sto_maml True \
    -sto_em True \
    -run_eval False \
    $@
