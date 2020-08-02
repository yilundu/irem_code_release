# Learning To Stop While Learning To Predict (ICML 2020)

- Authors: [Xinshi Chen](http://xinshi-chen.com/), [Hanjun Dai](https://hanjun-dai.github.io/), [Yu Li](https://liyu95.com), [Xin Gao](https://sfb.kaust.edu.sa/Pages/Home.aspx), [Le Song](https://www.cc.gatech.edu/~lsong/)

- [Link to paper](https://arxiv.org/abs/2006.05082)

- [Link to ICML 15-min presentation](https://icml.cc/virtual/2020/poster/6279)

If you found this library useful in your research, please consider citing

```
@article{chen2020learning,
  title={Learning to Stop While Learning to Predict},
  author={Chen, Xinshi and Dai, Hanjun and Li, Yu and Gao, Xin and Song, Le},
  journal={arXiv preprint arXiv:2006.05082},
  year={2020}
}
```


# Reproduce Experiments In Sec 5.1. Sparse Recovery

## Install the module
Please navigate to the root of this repository, and run the following command to install the `lista_stop` module.
```
pip install -e .
```

## Run traditional algorithms: ISTA and FISTA
Navigate to the `/lista_stop/baselines` folder and run the following commands to reproduce results of ISTA and FISTA, respectively.
```
sh run_ista.sh

sh run_fista.sh
```

## Run the baseline model: LISTA
Navigate to the `/lista_stop/experiments` folder and run the following command.
```
sh run_lista.sh
```

## Run our method: LISTA-stop
The training process of LISTA-stop has two stages. For stage 1 training, navigate to the `/lista_stop/experiments` folder and run the following command.
```
run_lista_stop_stage1.sh
```
For stage 2 training, run the following command.
```
run_lista_stop_stage2.sh
```

# Reproduce Experiments In Sec 5.2.

Coming soon.

# Reproduce Experiments In Sec 5.3.

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
```

## Deactivate the environment
```
source deactivate
```



