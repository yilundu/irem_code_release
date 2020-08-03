from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import os
import pickle as cp
import torch

cmd_opt = argparse.ArgumentParser(description='Argparser for metal optimal stopping')
cmd_opt.add_argument('-save_dir', default='.', help='result output root')
cmd_opt.add_argument('-data_root', default=None, help='dropbox folder')

cmd_opt.add_argument('-init_model_dump', default=None, help='model dump')
cmd_opt.add_argument('-epoch_load', default=None, type=int, help='epoch to load')

cmd_opt.add_argument('-data_name', default=None, help='dataset name')
cmd_opt.add_argument('-phase', default='train', help='phase')

cmd_opt.add_argument('-hidden_size', default=64, type=int, help='hidden dims')
cmd_opt.add_argument('-seed', default=19260817, type=int, help='seed')

cmd_opt.add_argument('-act_func', default='tanh', help='default activation function')

cmd_opt.add_argument('-num_train_shots', default=None, type=int, help='num shots in few shot learning training')
cmd_opt.add_argument('-min_train_shots', default=None, type=int, help='min num shots in few shot learning training')
cmd_opt.add_argument('-max_train_shots', default=None, type=int, help='max num shots in few shot learning training')

cmd_opt.add_argument('-num_test_shots', default=None, type=int, help='num shots in few shot learning testing')
cmd_opt.add_argument('-num_ways', default=5, type=int, help='num classes in few shot learning')

cmd_opt.add_argument('-num_data_workers', default=1, type=int, help='# data loading threads')
cmd_opt.add_argument('-gpu', default=0, type=int, help='index of gpu, if < 0 then use cpu')
cmd_opt.add_argument('-pos_dist', default='geo', help='posterior distribution, geo: geometric, multi: multinomial', choices=['geo', 'multi'])
cmd_opt.add_argument('-vi_net', default='outdiff', help='variational posterior parameterization', choices=['outdiff', 'shared'])
cmd_opt.add_argument('-sto_em', default=False, type=eval, help='stochastic EM?')


cmd_opt.add_argument('-lr_outer', default=1e-3, type=float, help='meta learning rate')
cmd_opt.add_argument('-entropy_penalty', default=1, type=float, help='entropy penalty coefficent')

cmd_opt.add_argument('-lr_inner', default=0.4, type=float, help='learning rate for inner update loop')
cmd_opt.add_argument('-grad_clip', default=5, type=float, help='clip gradient')
cmd_opt.add_argument('-dropout', default=0, type=float, help='dropout')
cmd_opt.add_argument('-num_unroll', default=1, type=int, help='# unrolling steps')
cmd_opt.add_argument('-num_test_unroll', default=None, type=int, help='# unrolling steps')

cmd_opt.add_argument('-first_order_only', default=False, type=eval, help='use only first order grad')
cmd_opt.add_argument('-run_eval', default=True, type=eval, help='run eval during train?')
cmd_opt.add_argument('-sto_maml', default=False, type=eval, help='sto maml?')
cmd_opt.add_argument('-grad_pos', default=False, type=eval, help='backprop through closed form posterior?')

cmd_opt.add_argument('-num_epochs', default=600, type=int, help='number of training epochs')
cmd_opt.add_argument('-batches_per_val', default=1000, type=int, help='number of batches per evaluation')
cmd_opt.add_argument('-samples_for_test', default=600, type=int, help='number of samples for evaluation')

cmd_opt.add_argument('-meta_batch_size', default=32, type=int, help='batch size (# tasks) for training')


cmd_args, _ = cmd_opt.parse_known_args()

if cmd_args.min_train_shots is None:
    cmd_args.min_train_shots = cmd_args.max_train_shots = cmd_args.num_train_shots

if cmd_args.num_train_shots is not None:
    assert cmd_args.min_train_shots == cmd_args.max_train_shots == cmd_args.num_train_shots

if cmd_args.num_test_shots is None:
    cmd_args.num_test_shots = cmd_args.num_train_shots

if cmd_args.num_test_unroll is None:
    cmd_args.num_test_unroll = cmd_args.num_unroll


if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

    if cmd_args.init_model_dump is None and cmd_args.epoch_load is not None:
        cmd_args.init_model_dump = os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % cmd_args.epoch_load)

if cmd_args.sto_maml:
    assert cmd_args.sto_em or cmd_args.entropy_penalty <= 0

print(cmd_args)


def set_device(gpu):
    if torch.cuda.is_available() and gpu >= 0:
        cmd_args.gpu = gpu
        cmd_args.device = torch.device('cuda:' + str(gpu))
        print('use gpu indexed: %d' % gpu)
    else:
        cmd_args.gpu = -1
        cmd_args.device = torch.device('cpu')
        print('use cpu')
