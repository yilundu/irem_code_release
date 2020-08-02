import argparse
import os

cmd_opt = argparse.ArgumentParser(description='DS')
cmd_opt.add_argument('-seed', type=int, default=0, help='seed')

cmd_opt.add_argument('-phase', type=str, default='train', help='training or testing phase')

# hyperparameters for training
cmd_opt.add_argument('-loss_type', type=str, default='mle', help='type of loss function')
cmd_opt.add_argument('-loss_weight', type=float, default=0.5)

cmd_opt.add_argument('-iters_per_eval', type=int, default=100, help='iterations per evaluation')
cmd_opt.add_argument('-iters_per_epoch', type=int, default=10, help='iterations per epoch')
cmd_opt.add_argument('-loss_temp', type=float, default=1, help='temperature for softmin in loss')
cmd_opt.add_argument('-min_temp', type=float, default=0, help='minimal temperature for softmin in loss')
cmd_opt.add_argument('-max_temp', type=float, default=0, help='maximal temperature for softmin in loss')

# hyperparameters for policy
cmd_opt.add_argument('-share', type=eval, default=True, help='pi_t for different t, share parameters or not')
cmd_opt.add_argument('-post_dim', type=int, default=8, help='position embedding dimension')
cmd_opt.add_argument('-val_policy_dump', type=str, default=None, help='best validation policy dump')
cmd_opt.add_argument('-stochastic', type=eval, default=True, help='stopping rule is stochastic or not')
cmd_opt.add_argument('-kl_type', type=str, default='forward', help='forward kl or backward kl')


cmd_opt.add_argument('-policy_type', type=str, default='sequential', help='choose from sequential and multiclass')
cmd_opt.add_argument("--outf", type=str, default="logs", help='path of log files')
cmd_opt.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
cmd_opt.add_argument('-batch_size', type=int, default=16, help='batch size')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-4, help='learning rate')
cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-temp', type=float, default=5, help='temperature for soft sign')

# hyperparameters for LISTA model
cmd_opt.add_argument('-T_max', type=int, default=50, help='max number of layers')
cmd_opt.add_argument('-num_output', type=int, default=10, help='num outputs')
cmd_opt.add_argument('-untied', type=eval, default=False, help='share parameters over layers or not')

cmd_opt.add_argument("--data_folder", type=str, default='./', help='the data folder')
cmd_opt.add_argument("--num_workers", type=int, default=12, help='the number of worker for io')
cmd_opt.add_argument("--restart", type=bool, default=False, help='load a snapshot and continue')

cmd_opt.add_argument("--val_ratio", type=float, default=0.2, help='the ratio for val data')


cmd_args = cmd_opt.parse_args()

print(cmd_args)