# pretrain the model
python -u train.py --model DnCNN --outf logs/dncnn_b_l20_all_train_n55 \
	--num_of_layers 20 --batchSize 256 --epoch 50

# fine-tuning with tao as 10
python -u train.py --model DnCNN_DS --outf logs/dncnn_b_ds_l20_all_train_tune_tao10 \
	--train_all True --batchSize 256 --lr 1e-4 --epoch 50 \
	--tao 10 --pretrain_path logs/dncnn_b_l20_all_train_n55/net.pth \
	--pretrain True

# policy training, this is in test phase
python train_stop_kl.py \
	--outf logs/dncnn_b_ds_l20_all_train_tune_tao10 --restart True -phase test

# joint training, this is in test phase
python train_stop_joint.py \
	--outf logs/dncnn_b_ds_l20_all_train_tune_tao10 --restart True -phase test

# Quantitative evaluation
for noise in 35 45 55 65 75; do
	python -u test.py --test_data Set68  --num_of_layers 20 \
		--logdir logs/dncnn_b_ds_l20_all_train_tune_tao10 --model DnCNN_DS \
		--test_noiseL ${noise}
done

# generate the denoised images
for noise in 45 65;do
	python -u test.py --test_data Set68  --num_of_layers 20 \
		--logdir logs/dncnn_b_ds_l20_all_train_tune_tao10 --model DnCNN_DS \
		--test_noiseL ${noise} --save_img True --img_folder ./out_imgs/dncnn_stop_${noise}
done