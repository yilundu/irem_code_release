from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import random
import numpy as np
import pickle
from lista_stop.common.utils import soft_threshold
from lista_stop.common.cmd_args import cmd_args
from lista_stop.dataset import ListaDataset


def MyMSE(x_hat, x):
    return torch.sum((x_hat - x) ** 2, dim=-1).view(-1)


class MetricEval(object):
    def __init__(self, dataset, test_key=None, test_snr=None):
        if test_snr is not None:
            y, x = dataset.static_data[test_key]
            self.mix = test_snr
        else:
            y, x = dataset.static_data['test']
            self.mix = dataset.mix
        self.y = y
        self.x = x
        self.A = dataset.A
        self.test_size = y.shape[0]
        self.denominator = torch.sum(x ** 2, dim=-1).mean()

        denominator_per_snr = []
        if self.mix is not None:
            size_per_snr = int(self.test_size / len(self.mix))
            self.size_per_snr = size_per_snr
            s = 0
            for i in range(len(self.mix)):
                denominator_per_snr.append(torch.sum(x[s:s+size_per_snr] ** 2, dim=-1).mean())
                s += size_per_snr
        self.denominator_per_snr = denominator_per_snr

    def compute(self, x_hat):
        mse = MyMSE(x_hat, self.x)
        mse_mean = mse.mean()
        nmse = 10 * torch.log10(mse_mean / self.denominator)
        mse_per_snr = []
        nmse_per_snr = []
        if self.mix is not None:
            s = 0
            for i in range(len(self.mix)):
                mse_per_snr.append(mse[s:s+self.size_per_snr].mean())
                s += self.size_per_snr
                nmse_per_snr.append(10 * torch.log10(mse_per_snr[i] / self.denominator_per_snr[i]))
        return mse_mean, nmse, mse_per_snr, nmse_per_snr


def ista_test(eval, algo_step, num_itr):
    with torch.no_grad():
        x_hat = torch.zeros(eval.x.shape)
        mse, nmse, mse_per_snr, nmse_per_snr = eval.compute(x_hat)
        print('inti nmse', nmse)
        nmse_dict = dict()
        for i in range(num_itr):
            x_hat = algo_step(eval.A, x_hat, eval.y)
            mse, nmse, mse_per_snr, nmse_per_snr = eval.compute(x_hat)
            nmse_dict[i] = nmse.cpu().item()
            print('%d samples, itr %d, mse: %.5f, nmse: %.5f'
                  % (eval.test_size, i+1, mse, nmse))
            print('nmse per snr', nmse_per_snr)
        return nmse_dict

def mse_loss(x_hat, x):
    return torch.sum((x_hat - x) ** 2)

def nmse_loss(x_hat, x):
    size = x_hat.shape[0]
    error = torch.sum((x_hat - x) ** 2, dim=-1)
    nm = torch.sum(x**2, dim=-1)
    assert nm.shape[0] == size
    return torch.sum(10 * torch.log10(error / nm))

class ISTA(object):
    def __init__(self, A, ld=0.05,  L=None):
        self.A = A
        if L > 0:
            self.L = L
        else:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            self.L = torch.max(s)
        self.ld = ld

    @staticmethod
    def ista_step(A, x, b, ld, L=None):
        if L is None:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            L = torch.max(s)
        g = x + A.t().dot(b - A.dot(x)) / L
        return soft_threshold(ld/L, g)

    @staticmethod
    def ista_step_batch(A, x, b, ld, L=None):
        """
        A: m * n
        x: size * n
        b: size * m
        """
        if L is None:
            u, s, v = torch.svd(torch.matmul(A.t(), A))
            L = torch.max(s)
        g = x + torch.matmul(b - torch.matmul(x, A.t()), A) / L
        return soft_threshold(ld/L, g)

    def ista(self, num_itr, A, x, b, batch=True):
        if batch:
            for _ in range(num_itr):
                x = self.ista_step_batch(A, x, b, self.ld, self.L)
        else:
            for _ in range(num_itr):
                x = self.ista_step(A, x, b, self.ld, self.L)
        return x


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

    algo_ista = ISTA(db.A, ld=cmd_args.rho, L=cmd_args.L)
    print(algo_ista.L)
    eval_class = MetricEval(db)
    nmse_dict = ista_test(eval_class, lambda a, x, y: algo_ista.ista_step_batch(a, x, y, algo_ista.ld, algo_ista.L), cmd_args.num_algo_itr)
    f = open('ISTA_converge.pkl', 'wb')
    pickle.dump(nmse_dict, f)
    f.close()
