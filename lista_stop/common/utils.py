from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from lista_stop.common.consts import DEVICE
from lista_stop.common.consts import NONLINEARITIES


def torch_rand_choice(choices, size):
    n = len(choices)
    idx = torch.randint(n, size)
    return choices[idx]


def position_encoding(n_pos, d):
    """
    position embedding in transformer
    :param n_pos: number of positions
    :param d: dimension of this embedding
    :return:
    """
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / d) for j in range(d)]
                            if pos != 0 else np.zeros(d) for pos in range(n_pos)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) # apply sin on 0th,2nd,4th...emb_dim
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) # apply cos on 1st,3rd,5th...emb_dim

    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def max_onehot(x, dim=-1):
    idx = torch.argmax(x, dim=dim)
    length = x.shape[dim]
    e = torch.eye(length).to(DEVICE)
    return e[idx.detach()]


def entropy(p):
    plogp = torch.where(p > 0, p * p.log(), p.new([0.0]))
    qlogq = torch.where(1-p > 0, (1-p) * (1-p).log(), p.new([0.0]))
    return -plogp-qlogq


def soft_threshold(theta, x):
    return torch.sign(x) * F.relu(torch.abs(x) - theta)


def soft_sign(x, k):
    return 2 * torch.sigmoid(k * x) - 1


def diff_soft_threshold(theta, x, k):
    return soft_sign(x, k) * F.relu(torch.abs(x) - theta)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, nonlinearity, act_last=None):
        super(MLP, self).__init__()
        self.act_last = act_last
        hidden_dims = tuple(map(int, hidden_dims.split("-")))
        prev_size = input_dim

        layers = []
        activation_fns = []
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            prev_size = h
            activation_fns.append(NONLINEARITIES[nonlinearity])
        if act_last is not None:
            activation_fns[-1] = NONLINEARITIES[self.act_last]
        self.output_size = prev_size
        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        weights_init(self)

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l + 1 < len(self.layers) or self.act_last is not None:
                x = self.activation_fns[l](x)
        return x


def glorot_uniform(t):
    if len(t.size()) == 2:
        fan_in, fan_out = t.size()
    elif len(t.size()) == 3:
        # out_ch, in_ch, kernel for Conv 1
        fan_in = t.size()[1] * t.size()[2]
        fan_out = t.size()[0] * t.size()[2]
    else:
        fan_in = np.prod(t.size())
        fan_out = np.prod(t.size())

    limit = np.sqrt(6.0 / (fan_in + fan_out))
    t.uniform_(-limit, limit)


def _param_init(m):
    if isinstance(m, Parameter):
        glorot_uniform(m.data)
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
        glorot_uniform(m.weight.data)


def weights_init(m):
    for p in m.modules():
        if isinstance(p, nn.ParameterList) or isinstance(p, nn.ModuleList):
            for pp in p:
                _param_init(pp)
        else:
            _param_init(p)

    for name, p in m.named_parameters():
        if not '.' in name: # top-level parameters
            _param_init(p)
