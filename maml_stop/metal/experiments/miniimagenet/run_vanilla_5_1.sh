#!/bin/bash

data_name=miniimagenet
data_root=../../../dataset
num_ways=5
num_train_shots=1
num_unroll=14
lr_inner=0.01
pos_dist=geo
vi_net=shared
grad_pos=False
entropy_penalty=0.001
sto_em=True

save_dir=../../../results/optstop-${data_name}-nw-${num_ways}-ns-${num_train_shots}-nr-${num_unroll}-pos-${pos_dist}-e-${entropy_penalty}-vi-${vi_net}-gpos-${grad_pos}-lr-${lr_inner}-se-${sto_em}

export CUDA_VISIBLE_DEVICES=0

python ../main_opt_stop.py \
    -data_root $data_root \
    -vi_net $vi_net \
    -sto_em $sto_em \
    -save_dir $save_dir \
    -grad_pos $grad_pos \
    -pos_dist $pos_dist \
    -data_name $data_name \
    -seed 10 \
    -num_ways $num_ways \
    -num_unroll $num_unroll \
    -num_train_shots $num_train_shots \
    -batches_per_val 100 \
    -lr_inner $lr_inner \
    -entropy_penalty $entropy_penalty \
    -meta_batch_size 4 \
    -num_epochs 600 \
    -num_test_shots 15 \
    -num_test_unroll $num_unroll \
    -run_eval False \
    -sto_maml True \
    $@
