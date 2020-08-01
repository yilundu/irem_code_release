#!/bin/bash
echo "script started"

phase=train
iters_per_eval=50
epoch=2000

dc=1e-4
batch_size=512

L=3
rho=0.1
num_layer=20
num_output=20
mix='20-30-40'

lr=1e-4

subdir=snr-${mix}-L-${num_layer}-output-${num_output}-batch-${batch_size}-L-${L}-rho-${rho}-epo-${epoch}-itr-${iters_per_eval}-lr-${lr}-dc-${dc}
echo "$subdir"
save_dir=../saved_model/lista/$subdir

model_dump=$save_dir/best_val_model.dump

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python3 lista.py \
    -loss_type sum \
    -gpu 0 \
    -snr_mix $mix \
    -batch_size $batch_size \
    -save_dir $save_dir \
    -iters_per_eval $iters_per_eval \
    -learning_rate $lr\
    -weight_decay $dc \
    -num_epochs $epoch \
    -T_max $num_layer \
    -num_output $num_output \
    -val_model_dump $model_dump \
    -phase $phase \
    -L $L \
    -rho $rho
echo "All scripts evaluated"
