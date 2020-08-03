from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import torch
import numpy as np


def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points
    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.
    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    with torch.no_grad():
        _, predictions = torch.max(logits, dim=-1)
        accuracy = torch.mean(predictions.eq(targets).float())
    return accuracy.item()


class OnlineStats(object):
    def __init__(self, init_func=lambda : 0, update_func=lambda x, y: x + y, readout_func=lambda x, y : x /y):
        super(OnlineStats, self).__init__()

        self.num_steps = 0
        self.update_func = update_func
        self.readout_func = readout_func
        self.init_func = init_func
        self.init()

    def init(self):
        self.stats = self.init_func()
        self.num_steps = 0

    def step(self, new_stat, n_step=1):
        self.num_steps += n_step
        self.stats = self.update_func(self.stats, new_stat)

    def summary(self):
        return self.readout_func(self.stats, self.num_steps)
