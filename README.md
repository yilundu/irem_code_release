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

Coming soon.


