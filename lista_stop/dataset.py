from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from lista_stop.common.consts import DEVICE
from lista_stop.common.utils import torch_rand_choice
import torch
import numpy as np
import numpy.linalg as la


class ListaDataset(object):
    def __init__(self, partition_sizes={}, m=250, n=500, con_num=5, pnz=0.1, snr=40.0, mix=None):
        """
        A: m * n matrix
        con_num: conditional number
        pnz: probability of non-zero
        snr: signal noise ratio
        """
        self.m = m
        self.n = n
        self.con_num = con_num
        self.pnz = pnz
        self.snr = snr
        self.A = torch.tensor(self.random_A(m, n, con_num)).to(DEVICE)
        if mix is not None:
            self.mix = list(map(int, mix.split("-")))
            self.snr = None
        else:
            self.mix = None
        self.static_data = {}
        for key in partition_sizes:
            num_data = partition_sizes[key]
            if key != 'train' and self.mix is not None:
                num_data = int(num_data / len(self.mix))
                y = [[] for _ in range(len(self.mix))]
                x = [[] for _ in range(len(self.mix))]
                for i in range(len(self.mix)):
                    y[i], x[i] = self.get_samples(num_data, self.mix[i], mix=None)
                self.static_data[key] = [torch.cat(y, dim=0), torch.cat(x, dim=0)]
            else:
                self.static_data[key] = self.get_samples(num_data, self.snr, self.mix)

    @staticmethod
    def random_A(M, N, con_num=0, col_normalized=True):
        """
        Randomly sample measurement matrix A.
        Curruently I sample A from i.i.d Gaussian distribution with 1./M variance and
        normalize columns.
        :M: integer, dimension of measurement y
        :N: integer, dimension of sparse code x
        :col_normalized:
            boolean, indicating whether normalize columns, default to True
        :returns:
            A: numpy array of shape (M, N)
        """
        A = np.random.normal( scale=1.0/np.sqrt(M), size=(M,N) ).astype(np.float32)
        if con_num > 0:
            U, _, V = la.svd (A, full_matrices=False)
            s = np.logspace (0, np.log10 (1 / con_num), M)
            A = np.dot (U * (s * np.sqrt(N) / la.norm(s)), V).astype (np.float32)
        if col_normalized:
            A = A / np.sqrt (np.sum (np.square (A), axis=0, keepdims=True))
        return A

    def get_samples(self, size, snr, mix):
        """
        Generate samples (y, x) in current problem setting.
        return y: size * n
        return x: size * n
        """
        bernoulli = (torch.rand([self.n, size]).to(DEVICE) <= self.pnz).float()
        x = bernoulli * torch.randn([self.n, size]).to(DEVICE)

        y = self.linear_measure(x, snr, mix)
        return y.t(), x.t()

    def gen_samples(self, size):
        """
        Generator
        """
        while 1:
            yield self.get_samples(size, snr=self.snr, mix=self.mix)

    def linear_measure(self, x, snr=None, mix=None):
        """
                Measure sparse code x with matrix A and return the measurement.
                x: n * batch_size
                A: m * n
                return y: m * batch_size
        """
        if mix is None:
            if snr is None:
                snr = self.snr
        else:
            snr = torch_rand_choice(torch.tensor(mix).to(DEVICE), size=x.shape[1:2]).view(-1)

        y = torch.matmul(self.A, x)
        std = torch.std(y, dim=0) * torch.pow(10.0, torch.tensor(-snr/20.0).float().to(DEVICE))
        std = torch.max(std, torch.tensor(10e-50).to(DEVICE))
        noise = torch.randn(size=y.shape).to(DEVICE) * std

        return y + noise

    def load_data(self, batch_size, phase, auto_reset=True, shuffle=True):

        if phase != 'train':
            assert not auto_reset
            assert phase in self.static_data

        if phase in self.static_data:  # generate mini-batches from dataset
            data_y, data_x = self.static_data[phase]
            data_size = data_y.shape[0]
            while True:
                if shuffle:
                    perms = torch.randperm(data_size)
                    data_y = data_y[perms, :]
                    data_x = data_x[perms, :]
                for pos in range(0, data_size, batch_size):
                    if pos + batch_size > data_size:  # the last mini-batch has fewer samples
                        if auto_reset:  # no need to use this last mini-batch
                            break
                        else:
                            num_samples = data_size - pos
                    else:
                        num_samples = batch_size

                    yield data_y[pos : pos + num_samples, :], data_x[pos : pos + num_samples, :]
                if not auto_reset:
                    break
        else:
            yield self.get_samples(batch_size, self.snr, self.mix)
