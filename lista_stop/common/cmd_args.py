import argparse
import os
from lista_stop.common.consts import NONLINEARITIES

cmd_opt = argparse.ArgumentParser(description='LISTA-stop')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')
cmd_opt.add_argument('-num_test', type=int, default=3000, help='number of test samples')
cmd_opt.add_argument('-num_val', type=int, default=3000, help='number of validation samples')
cmd_opt.add_argument('-num_val_policy', type=int, default=10000, help='number of validation samples for policy network')
cmd_opt.add_argument('-seed', type=int, default=19260817, help='seed')

cmd_opt.add_argument('-phase', type=str, default='train', help='training or testing phase')

# hyperparameters for training
cmd_opt.add_argument('-loss_type', type=str, default='mle', help='type of loss function')
cmd_opt.add_argument('-loss_weight', type=float, default=0.5)

cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
cmd_opt.add_argument('-learning_rate2', type=float, default=1e-3, help='learning rate for policy during joint training')

cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-iters_per_eval', type=int, default=100, help='iterations per evaluation')
cmd_opt.add_argument('-iters_per_epoch', type=int, default=100, help='iterations per epoch')
cmd_opt.add_argument('-loss_temp', type=float, default=1, help='temperature for softmin in loss')
cmd_opt.add_argument('-min_temp', type=float, default=0, help='minimal temperature for softmin in loss')
cmd_opt.add_argument('-max_temp', type=float, default=0, help='maximal temperature for softmin in loss')

cmd_opt.add_argument('-init_model_dump', type=str, default=None, help='model initilization dump')
cmd_opt.add_argument('-val_model_dump', type=str, default=None, help='best validation model dump')

cmd_opt.add_argument('-var', type=float, default=1, help='variance of Gaussian distribution')

# hyperparameters for policy
cmd_opt.add_argument('-share', type=eval, default=True, help='pi_t for different t, share parameters or not')
cmd_opt.add_argument('-post_dim', type=int, default=8, help='position embedding dimension')
cmd_opt.add_argument('-entropy-coef', type=float, default=0.1, help='probability of non-zero')

cmd_opt.add_argument('-policy_type', type=str, default='sequential', help='choose from sequential and multiclass')

cmd_opt.add_argument('-val_policy_dump', type=str, default=None, help='best validation policy dump')
cmd_opt.add_argument('-policy_hidden_dims', type=str, default='512-256-128-64-1', help='policy MLP hidden sizes')
cmd_opt.add_argument('-policy_multiclass_dims', type=str, default='64-64', help='classification output layers')

cmd_opt.add_argument('-nonlinearity', type=str, default='relu', choices=NONLINEARITIES)
cmd_opt.add_argument('-stochastic', type=eval, default=True, help='stopping rule is stochastic or not')
cmd_opt.add_argument('-kl_type', type=str, default='forward', help='forward kl or backward kl')


# hyperparameters for ISTA
cmd_opt.add_argument('-num_algo_itr', type=int, default=50, help='number of iterations in the algorithm')
cmd_opt.add_argument('-L', type=float, default=0, help='inverse of step size')

# hyperparameters for LISTA model
cmd_opt.add_argument('-T_max', type=int, default=50, help='max number of layers')
cmd_opt.add_argument('-num_output', type=int, default=10, help='num outputs')

# LASSO Problem configuration
# A: m * n matrix
# x: n vector

cmd_opt.add_argument('-m', type=int, default=250, help='dim1 of A')
cmd_opt.add_argument('-n', type=int, default=500, help='dim2 of A')
cmd_opt.add_argument('-rho', type=float, default=0.2, help='sparsity coefficient')
cmd_opt.add_argument('-pnz', type=float, default=0.1, help='probability of non-zero')
cmd_opt.add_argument('-snr', type=float, default=40, help='signal noise ratio')
cmd_opt.add_argument('-snr_mix', type=str, default=None, help='mix signal noise ratio. eg: 20-30-40')
cmd_opt.add_argument('-test_snr', type=str, default=None, help='set test mix signal noise ratio. eg: 5-10')


cmd_opt.add_argument('-con_num', type=float, default=5, help='condition number of matrix A')

cmd_opt.add_argument('-temp', type=float, default=5, help='temperature for soft sign')

cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
