
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import sys
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random

from metal.common.cmd_args import cmd_args
from metal.common.meta_utils import maml_step, sto_maml_step

from metal.experiments.utils import get_accuracy


def detach_list(l):
    return [x.detach() for x in l]


def opt_stop_task(model, vi_net, train_input, train_target, test_input, test_target, num_unroll, phase):
    loss_func = F.cross_entropy
    if cmd_args.sto_maml:
        maml_func = sto_maml_step
    else:
        maml_func = maml_step

    inner_losses, list_outer_loss, train_pred_list, test_pred_list, train_feat_list, test_feat_list, param_list =  maml_func(model, loss_func=F.cross_entropy,
                batch_data=(train_input, train_target, test_input, test_target),
                lr_inner=cmd_args.lr_inner,
                num_unroll=num_unroll,
                first_order=cmd_args.first_order_only or phase != 'train')
    outer_losses = torch.stack(list_outer_loss).view(-1)
    if cmd_args.entropy_penalty > 0:
        posterior = torch.exp(F.log_softmax(-outer_losses / cmd_args.entropy_penalty, dim=0))
        if not cmd_args.grad_pos or cmd_args.sto_em:
            posterior.detach()
        if cmd_args.sto_em:
            t_idx = torch.multinomial(posterior, 1).item()
            loss = list_outer_loss[t_idx]            
        else:
            loss = torch.sum(posterior * outer_losses)
    else:
        t_idx = torch.argmin(outer_losses).item()
        loss = list_outer_loss[t_idx]

    if cmd_args.sto_maml:
        _, test_pred = model(test_input, params=param_list[t_idx])
        loss = loss_func(test_pred, test_target)

    log_pi, best_pos = vi_net(detach_list(inner_losses), 
                    detach_list(train_feat_list), 
                    detach_list(test_feat_list), 
                    detach_list(train_pred_list), 
                    train_input,
                    train_target,
                    test_input)
    with torch.no_grad():
        acc = get_accuracy(test_pred_list[best_pos], test_target)

    if phase == 'train':
        t_map = torch.argmin(outer_losses).item()
        log_pi = log_pi.view(-1)
        ce = -log_pi[t_map]
        loss = loss + ce
        return loss, (acc, ce.item())
    elif phase == 'val':
        return loss, acc
    else:
        t_true = torch.argmin(outer_losses).item()
        with torch.no_grad():
            best_acc = get_accuracy(test_pred_list[t_true], test_target)

        posterior = torch.exp(F.log_softmax(-outer_losses / cmd_args.entropy_penalty, dim=0)).detach().view(-1).cpu().data.numpy()
        vi = torch.exp(log_pi).detach().view(-1).data.cpu().numpy()
        return loss, (acc, best_pos, t_true, best_acc, vi, posterior)


def train_callback(batch_idx, outer_loss, list_out, stats):
    list_acc = [x[0] for x in list_out]
    list_ce = [x[1] for x in list_out]
    acc = np.mean(list_acc)
    ce = np.mean(list_ce)
    stats.step(np.array([acc * len(list_acc), ce * len(list_acc)]), n_step=len(list_acc))
    msg = 'train batch acc: %.2f, ce: %.4f' % (acc * 100, ce)
    return msg


def val_callback(batch_idx, outer_loss, list_acc):
    acc = np.mean(list_acc)
    msg = 'val batch acc: %.2f, val outer_loss: %.2f' % (acc, outer_loss)
    return msg


def test_callback(batch_idx, outer_loss, list_out, acc_stats):
    list_acc = [x[0] for x in list_out]    
    stat = np.zeros(cmd_args.num_test_unroll * 2 + 3 + cmd_args.num_test_unroll * 2, dtype=np.float32)
    pos = cmd_args.num_test_unroll * 2 + 3
    acc2 = np.sum([x**2 for x in list_acc])
    best_acc = np.sum([x[3] for x in list_out])
    for x in list_out:
        stat[x[1] + 1] += 1
        stat[cmd_args.num_test_unroll + 2 + x[2]] += 1
        stat[pos : pos + cmd_args.num_test_unroll] += x[4]
        stat[pos + cmd_args.num_test_unroll : ] += x[5]
    acc = np.sum(list_acc)
    stat[0] = acc
    stat[cmd_args.num_test_unroll + 1] = acc2
    stat[pos - 1] = best_acc

    acc_stats.step(stat, n_step=len(list_out))
    avg_stats = acc_stats.summary()
    msg = 'test total acc: %.4f' % avg_stats[0]
    for i in range(0, cmd_args.num_test_unroll, 3):
        msg += ', #%d: %.2f' % (i + 1, avg_stats[i + 1])
    return msg
