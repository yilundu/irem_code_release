#!/bin/bash

data_name=miniimagenet
data_root=../../../dataset
num_ways=5
min_train=1
max_train=10
num_unroll=10
num_test=5
lr_inner=0.01
lr_outer=0.0001
pos_dist=geo
vi_net=shared
grad_pos=False
entropy_penalty=0.01
sto_em=True
sto_maml=True

save_dir=../../../results/metal-imba/optstop-${data_name}-nw-${num_ways}-ns-${min_train}-${max_train}-nr-${num_unroll}-pos-${pos_dist}-e-${entropy_penalty}-vi-${vi_net}-gpos-${grad_pos}-lr-${lr_inner}-se-${sto_em}-sm-${sto_maml}-lo-${lr_outer}-test-${num_test}
export CUDA_VISIBLE_DEVICES=0

python ../main_opt_stop.py \
    -data_root $data_root \
    -vi_net $vi_net \
    -sto_em $sto_em \
    -save_dir $save_dir \
    -grad_pos $grad_pos \
    -pos_dist $pos_dist \
    -data_name $data_name \
    -num_ways $num_ways \
    -num_unroll $num_unroll \
    -min_train_shots $min_train \
    -max_train_shots $max_train \
    -batches_per_val 100 \
    -lr_inner $lr_inner \
    -entropy_penalty $entropy_penalty \
    -meta_batch_size 1 \
    -num_epochs 600 \
    -num_test_shots $num_test \
    -num_test_unroll $num_unroll \
    -sto_maml $sto_maml \
    -lr_outer $lr_outer \
    -run_eval False \
    $@
