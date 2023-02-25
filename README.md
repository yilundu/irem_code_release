
# Reproduce Experiments In Sec 5.1. Sparse Recovery

## Install the module
Please navigate to the root of this repository, and run the following command to install the `lista_stop` module.
```
pip install -e .
```

# Reproduce Image Denoising

## Configure the environment
Please navigate to the section folder `./dncnn_stop`. Then, using the following command, we can configure the environment for the denoise experiments. Please keep the environment activated for this section.
```
conda env create -f environment.yml
source activate dncnn_stop
```

## Download the dataset
Please download the [dataset](https://www.dropbox.com/s/95xkvazbspwvury/data.zip?dl=0) and unzip the dataset in this folder.

## Run our method
Please run the following command to check our method.
```
# pretrain the model
python -u train.py --model ebm --outf logs/dncnn_b_l20_all_train_n55 \
	--num_of_layers 20 --batchSize 256 --epoch 50

# generate the denoised images
for noise in 45 65;do
	python -u test.py --test_data Set68  --num_of_layers 20 \
		--logdir logs/dncnn_b_ds_l20_all_train_tune_tao10 --model ebm \
		--test_noiseL ${noise} --save_img True --img_folder ./out_imgs/dncnn_stop_${noise}
done
