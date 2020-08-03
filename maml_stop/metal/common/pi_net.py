from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from tqdm import tqdm
import numpy as np
import random
from metal.common.torch_utils import MLP


class Conv2d(nn.Conv2d):
    __doc__ = nn.Conv2d.__doc__

    def forward(self, input):        
        params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            params['weight'], bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, params['weight'], bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv_block(in_channels, out_channels, **kwargs):
    return nn.Sequential(OrderedDict([
        ('conv', Conv2d(in_channels, out_channels, **kwargs)),
        ('norm', nn.BatchNorm2d(out_channels)),
        ('relu', nn.ReLU()),
        ('pool', nn.MaxPool2d(2))
    ]))


class Cnn4(nn.Module):
    def __init__(self, in_channels, hidden_size=64):
        super(Cnn4, self).__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size

        self.features = nn.Sequential(OrderedDict([
            ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True)),
            ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                                  stride=1, padding=1, bias=True))
        ]))

    def forward(self, inputs):
        features = self.features(inputs)
        return features.view((features.size(0), -1))


class DeepSets(nn.Module):
    def __init__(self, input_dim, out_dim, n_hidden=1, act_func=F.elu):
        super(DeepSets, self).__init__()
        self.act_func = act_func
        self.input_dim = input_dim
        self.out_feat = MLP(input_dim, [out_dim] * n_hidden)

    def forward(self, features, n_groups=1):
        features = F.softplus(features)
        assert features.shape[0] % n_groups == 0
        features = features.view(n_groups, -1, self.input_dim)

        flat_feat = torch.mean(features, dim=1)

        out_feat = self.out_feat(flat_feat)
        return out_feat


def get_geo_log_posterior(scores):
    prob = F.sigmoid(scores)
    np_prob = prob.view(-1).data.cpu().numpy()
    neg_prob = 1.0 - prob

    prefix_sum = scores.new(scores.shape[0], 1).zero_()
    list_sum = [prefix_sum]
    best_pos = None
    for i in range(1, scores.shape[1]):
        prefix_sum = prefix_sum + torch.log(neg_prob[:, i].view(-1, 1) + 1e-32)
        if np_prob[i - 1] > 0.5 and best_pos is None:
            best_pos = i - 1
        list_sum.append(prefix_sum)
    if best_pos is None:
        best_pos = scores.shape[1] - 1
    last_prob = prob.new(scores.shape[0], 1).zero_() + 1
    prob = torch.cat([prob[:, :-1], last_prob], dim=-1)
    prefix_sum = torch.cat(list_sum, dim=-1)
    return torch.log(prob + 1e-32) + prefix_sum, best_pos


def get_multinomial_log_posterior(scores):
    return F.log_softmax(scores, dim=-1), torch.argmax(scores).item()


class VICondNet(nn.Module):
    def __init__(self, args):
        super(VICondNet, self).__init__()
        self.hidden_size = args.hidden_size
        self.out_dim = args.num_ways

        self.target_encoder = MLP(self.out_dim, [self.hidden_size] * 2)

        if args.pos_dist == 'geo':
            self.pos_func = get_geo_log_posterior
        elif args.pos_dist == 'multi':
            self.pos_func = get_multinomial_log_posterior
        else:
            raise NotImplementedError

    def get_pred_target_diff(self, train_target, train_input, list_preds):
        train_target = train_target.view(train_input.shape[0], -1)

        if train_target.shape[1] < self.out_dim:
            train_target = train_input.new(train_input.shape[0], self.out_dim).zero_().scatter(1, train_target, 1)

        y_train = self.target_encoder(train_target)        
        y_pred = torch.stack(list_preds).view(-1, self.out_dim)
        y_pred = self.target_encoder(y_pred)
        
        diff = y_pred - y_train.repeat(len(list_preds), 1)
        return diff

class ShareFeatCondNet(VICondNet):
    def __init__(self, args):
        super(ShareFeatCondNet, self).__init__(args)

        self.feat_set = DeepSets(self.hidden_size, self.hidden_size)
        self.target_encoder = MLP(self.out_dim, [self.hidden_size] * 2)
        self.target_set = DeepSets(self.hidden_size, self.hidden_size)
        self.score_func = MLP(self.hidden_size * 3 + 1, [self.hidden_size * 2, 1])

    def forward(self, list_losses, list_train_feat, list_test_feat, list_preds, train_input, train_target, test_input):
        x_train = [self.feat_set(x) for x in list_train_feat]
        x_test = [self.feat_set(x) for x in list_test_feat]

        x_train = torch.cat(x_train, dim=0)
        x_test = torch.cat(x_test, dim=0)
        diff = self.get_pred_target_diff(train_target, train_input, list_preds)
        diff_feat = self.target_set(diff, n_groups=len(list_preds))
        losses = torch.stack(list_losses).view(-1, 1)
        final_feat = torch.cat([x_train, x_test, diff_feat, losses], dim=-1)
    
        scores = self.score_func(final_feat)
        return self.pos_func(scores.view(1, -1))

class OutDiffCondNet(VICondNet):
    def __init__(self, args):
        super(OutDiffCondNet, self).__init__(args)

        if args.data_name == 'omniglot':
            self.input_net = Cnn4(1, hidden_size=self.hidden_size)            
        elif args.data_name == 'miniimagenet':
            self.input_net = Cnn4(3, hidden_size=self.hidden_size)            
        else:
            raise NotImplementedError
        
        self.feat_set = DeepSets(self.hidden_size, self.hidden_size)        
        self.target_set = DeepSets(self.hidden_size, self.hidden_size)
        self.score_func = MLP(self.hidden_size * 3 + 1, [self.hidden_size * 2, 1])

    def forward(self, list_losses, list_train_feat, list_test_feat, list_preds, train_input, train_target, test_input):
        x_train = self.feat_set(self.input_net(train_input))
        x_test = self.feat_set(self.input_net(test_input))

        diff = self.get_pred_target_diff(train_target, train_input, list_preds)
        diff_feat = self.target_set(diff, n_groups=len(list_preds))
        losses = torch.stack(list_losses).view(-1, 1)
        
        x_train = x_train.repeat(losses.shape[0], 1)
        x_test = x_test.repeat(losses.shape[0], 1)
        final_feat = torch.cat([x_train, x_test, diff_feat, losses], dim=-1)
    
        scores = self.score_func(final_feat)
        return self.pos_func(scores.view(1, -1))

