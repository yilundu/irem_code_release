from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np
import math

from lista_stop.common.utils import soft_threshold
from lista_stop.common.cmd_args import cmd_args
from lista_stop.dataset import ListaDataset
from lista_stop.baselines.ista import MetricEval
import pickle


def fista_test(eval, algo_step, num_itr):
    with torch.no_grad():
        x_hat = torch.zeros(eval.x.shape)
        y_hat = torch.zeros(eval.x.shape)
        t = 1.0
        nmse_dict = dict()
        for i in range(num_itr):
            x_hat, y_hat, t = algo_step(eval.A, x_hat, y_hat, eval.y, t)
            mse, nmse, mse_per_snr, nmse_per_snr = eval.compute(y_hat)
            nmse_dict[i] = nmse.cpu().item()
            print('%d samples, itr %d, mse: %.5f, nmse: %.5f'
                  % (eval.test_size, i+1, mse, nmse))
            print('nmse per snr', nmse_per_snr)
        return nmse_dict


class FISTA(object):
    def __init__(self, A, ld=0.05,  L=None):
        self.A = A
        if L > 0:
            self.L = L
        else:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            self.L = torch.max(s)
        self.ld = ld
        self.t = 1.0

    @staticmethod
    def fista_step(A, xk_1, yk, b, ld, t, L=None):
        if L is None:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            L = torch.max(s)
        g = yk + A.t().dot(b - A.dot(yk)) / L
        x = soft_threshold(ld/L, g)
        t_next = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
        y = x + (t - 1) / t_next * (x - xk_1)
        return x, y

    @staticmethod
    def fista_step_batch(A, xk_1, yk, b, ld, t, L=None):
        """
        A: m * n
        x: size * n
        b: size * m
        """
        if L is None:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            L = torch.max(s)
        g = yk + torch.matmul(b - torch.matmul(yk, A.t()), A) / L
        x = soft_threshold(ld/L, g)
        t_next = (1 + math.sqrt(1 + 4 * t ** 2)) / 2
        y = x + (t - 1) / t_next * (x - xk_1)
        return x, y, t_next


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # initilize validation and test data
    db = ListaDataset(partition_sizes={'test': cmd_args.num_test,
                                       'val': cmd_args.num_val},
                      m=cmd_args.m,
                      n=cmd_args.n,
                      con_num=cmd_args.con_num,
                      pnz=cmd_args.pnz,
                      snr=cmd_args.snr,
                      mix=cmd_args.snr_mix)

    print('initialize the matrices A of size = ', cmd_args.m, cmd_args.n)

    if cmd_args.snr_mix is not None:
        mix = list(map(int, cmd_args.snr_mix.split("-")))
        n_data_per_snr = int(cmd_args.num_test / len(mix))

    algo_fista = FISTA(db.A, ld=cmd_args.rho, L=cmd_args.L)
    eval_class = MetricEval(db)
    nmse_dict = fista_test(eval_class, lambda a, x, y, b, t: algo_fista.fista_step_batch(a, x, y, b, algo_fista.ld, t, algo_fista.L), cmd_args.num_algo_itr)
    f = open('FISTA_converge.pkl', 'wb')
    pickle.dump(nmse_dict, f)
    f.close()
