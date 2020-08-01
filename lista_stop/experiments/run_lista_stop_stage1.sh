#!/bin/bash
echo "script started"

phase=train

lr=1e-4
dc=1e-4

L=3
rho=0.1
var=0.14
num_layer=20
num_output=20
mix='20-30-40'

iters_per_eval=25
epoch=2000
batch_size=1024

subdir=snr-${mix}-L-${num_layer}-output-${num_output}-batch-${batch_size}-L-${L}-rho-${rho}-var-${var}-epo-${epoch}-itr-${iters_per_eval}-lr-${lr}-dc-${dc}
echo "$subdir"
save_dir=../saved_model/lista/stage1/$subdir
model_dump=$save_dir/best_val_model.dump

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

python3 lista.py \
    -loss_type mle \
    -gpu 1 \
    -snr_mix $mix \
    -batch_size $batch_size \
    -save_dir $save_dir \
    -iters_per_eval $iters_per_eval \
    -learning_rate $lr\
    -var $var \
    -weight_decay $dc \
    -num_epochs $epoch \
    -T_max $num_layer \
    -num_output $num_output \
    -val_model_dump $model_dump \
    -phase $phase \
    -L $L \
    -rho $rho

echo "All scripts evaluated"
