import torch.nn as nn
import torch
import numpy as np
from torch.nn.parameter import Parameter
from lista_stop.common.consts import DEVICE

from lista_stop.common.utils import diff_soft_threshold, MLP, position_encoding


class LISTA(nn.Module):
    def __init__(self, A, args):
        """
        :param A: m*n matrix
        :param self.T_max: maximal number of layers
        :param self.num_output: number of output layers
        :param self.ld: initial weight of the sparsity coefficient
        """
        super(LISTA, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        # Initialize the parameters
        if args.L > 0:
            self.L = torch.tensor(args.L)
        else:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            self.L = torch.max(s)
        self.theta = 1/self.L * args.rho
        self.Wx = torch.eye(self.n).to(DEVICE) - torch.matmul(self.A.t(), self.A) / self.L
        self.Wb = self.A.t() / self.L
        self.T_max = args.T_max
        self.num_output = args.num_output
        self.k = args.temp
        self.ld = args.rho

        linear_x_t = []
        linear_b_t = []
        theta_t = []
        for t in range(self.T_max):
            linear_x_t.append(nn.Linear(self.n, self.n, bias=False))
            linear_b_t.append(nn.Linear(self.m, self.n, bias=False))
            theta_t.append(Parameter(self.theta))
        self.linear_x_t = nn.ModuleList(linear_x_t)
        self.linear_b_t = nn.ModuleList(linear_b_t)
        self.theta_t = nn.ParameterList(theta_t)
        # weight initialization
        with torch.no_grad():
            for t in range(self.T_max):
                self.linear_x_t[t].weight.copy_(self.Wx)
                self.linear_b_t[t].weight.copy_(self.Wb)

    def forward(self, y, x0=None):

        if x0 is None:
            batch_size = y.shape[0]
            xh = torch.zeros([batch_size, self.n]).to(DEVICE)
        else:
            xh = x0.clone()

        xhs_ = []
        r = np.remainder(self.T_max, self.num_output)
        d = np.floor_divide(self.T_max, self.num_output)
        out_idx = r+d-1

        for t in range(self.T_max):

            g = self.linear_b_t[t](y) + self.linear_x_t[t](xh)
            # The soft-threshold operation is not differentiable.
            # Here we use a smoothed version of soft-threshold so that it can be differentiable.
            # k is the temperature which can be tuned.
            xh = diff_soft_threshold(self.theta_t[t], g, self.k)

            if t == out_idx:
                xhs_.append(xh)
                out_idx += d
        assert self.T_max == out_idx - d + 1

        return xhs_


# Architecture of the stopping policy network.
class SeqNet(nn.Module):
    def __init__(self, A, args, train_post):
        super(SeqNet, self).__init__()
        self.A = A
        self.m, self.n = A.shape
        self.args = args
        hidden_dims = args.policy_hidden_dims + '-' + str(1) # output size is one (score)
        self.mlp = MLP(input_dim=2*self.m + self.n + args.post_dim,
                       hidden_dims=hidden_dims,
                       nonlinearity=args.nonlinearity,
                       act_last=None)
        self.post_dim = args.post_dim
        self.train_post = train_post

    # it returns the logits instead of the probability.
    def forward(self, y, xhs):
        pi_t_score = []
        position_enc = position_encoding(len(self.train_post), self.post_dim).to(DEVICE)
        batch_size = y.shape[0]
        for i, t in self.train_post.items():
            x_hat = xhs[t]
            ax = torch.matmul(x_hat, self.A.t())
            pe = torch.stack([position_enc[i] for _ in range(batch_size)], dim=0)
            f = torch.cat([y, y-ax, x_hat, pe], dim=-1)
            pi_t_score.append(self.mlp(f))

        scores = torch.cat(pi_t_score, dim=-1)
        return scores
