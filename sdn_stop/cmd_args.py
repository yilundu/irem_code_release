import argparse
import os

cmd_opt = argparse.ArgumentParser(description='Dynamic stopping for SDN')
cmd_opt.add_argument('-save_dir', type=str, default='./policy_networks', help='save folder')
# cmd_opt.add_argument('-num_test', type=int, default=1000, help='number of test samples')
# cmd_opt.add_argument('-num_val', type=int, default=1000, help='number of validation samples')
# cmd_opt.add_argument('-num_val_policy', type=int, default=10000, help='number of validation samples for policy network')
cmd_opt.add_argument('-seed', type=int, default=0, help='seed')

cmd_opt.add_argument('-phase', type=str, default='train', help='training or testing phase')

# hyperparameters for training
cmd_opt.add_argument('-loss_type', type=str, default='mle', help='type of loss function')
cmd_opt.add_argument('-loss_weight', type=float, default=0.5)

cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-learning_rate', type=float, default=1e-3, help='learning rate')
cmd_opt.add_argument('-weight_decay', type=float, default=0)
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-iters_per_eval', type=int, default=100, help='iterations per evaluation')
cmd_opt.add_argument('-iters_per_epoch', type=int, default=100, help='iterations per epoch')
cmd_opt.add_argument('-loss_temp', type=float, default=1, help='temperature for softmin in loss')
cmd_opt.add_argument('-min_temp', type=float, default=0, help='minimal temperature for softmin in loss')
cmd_opt.add_argument('-max_temp', type=float, default=0, help='maximal temperature for softmin in loss')

cmd_opt.add_argument('-init_model_dump', type=str, default=None, help='model initilization dump')
cmd_opt.add_argument('-val_model_dump', type=str, default=None, help='best validation model dump')

# cmd_opt.add_argument('-var', type=float, default=1, help='variance of Gaussian distribution')

cmd_opt.add_argument('-num_output', type=int, default=7, help='number of outputs')

# hyperparameters for policy
cmd_opt.add_argument('-share', type=eval, default=True, help='pi_t for different t, share parameters or not')
cmd_opt.add_argument('-post_dim', type=int, default=8, help='position embedding dimension')

cmd_opt.add_argument('-policy_type', type=str, default='sequential', 
	help='choose from sequential and multiclass')
cmd_opt.add_argument('-model_type', type=str, default='imiconfidence', 
	help='choose from sequential and multiclass, confidence, imiconfidence')
cmd_opt.add_argument('-net_size', type=int, default=3, help='the network size ratio')

cmd_opt.add_argument('-val_policy_dump', type=str, default=None, help='best validation policy dump')
cmd_opt.add_argument('-policy_multiclass_dims', type=str, default='64-64', help='classification output layers')

cmd_opt.add_argument('-nonlinearity', type=str, default='relu')
cmd_opt.add_argument('-stochastic', type=eval, default=False, help='stopping rule is stochastic or not')
cmd_opt.add_argument('-kl_type', type=str, default='forward', help='forward kl or backward kl')

cmd_opt.add_argument('-temp', type=float, default=5, help='temperature for soft sign')

cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
