from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lista_stop.common.consts import DEVICE
from lista_stop.common.cmd_args import cmd_args
from lista_stop.dataset import ListaDataset
from lista_stop.model import LISTA
from lista_stop.trainer import ListaModelMLE
from lista_stop.baselines.ista import MetricEval
import torch.optim as optim

import torch
import random
import numpy as np
from lista_stop.trainer import MyMSE
import pickle


def test(net, data_loader):
    net.eval()
    y, x = data_loader.static_data['test']
    # error
    with torch.no_grad():
        xhs = net(y)
        # loss of last layer
        x_hat = xhs[-1]
        loss_last = MyMSE(x_hat, x).mean()
        # val loss of best layer
        loss_t = []
        loss_per_layer = []
        for t in range(net.num_output):
            x_hat = xhs[t]
            mse = MyMSE(x_hat, x)
            loss_t.append(mse)
            loss_per_layer.append(mse.mean())
        loss_best, indices = torch.min(torch.stack(loss_t, dim=0), dim=0)
        loss_best = loss_best.mean()

        log = dict()
        best_distribution = []
        for t in range(net.num_output):
            key = 'loss of layer %d' % (t+1)
            log[key] = loss_per_layer[t]
            best_distribution.append(int((indices == t).sum()))

        log['best layer distribution'] = best_distribution

        log['loss of last layer'] = loss_last
        log['loss of best layer'] = loss_best

        return log


def lista_test(eval, net):
    net.eval()
    # error
    with torch.no_grad():
        xhs = net(eval.y)
        nmse_dict = dict()
        for t in range(net.num_output):
            x_hat = xhs[t]
            mse, nmse, mse_per_snr, nmse_per_snr = eval.compute(x_hat)
            nmse_dict[t] = nmse.cpu().item()
            print('%d samples, %d-th output, mse: %.5f, nmse: %.5f'
                  % (eval.test_size, t+1, mse, nmse))
            nmse_print = 'nmse per snr'
            for nmse in nmse_per_snr:
                nmse_print += ', %.5f' % nmse
            print(nmse_print)
        return nmse_dict


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)
    # initialize validation and test data at the beginning.
    # since the random seed is set, the validation and test data will be the same for all programs.
    db = ListaDataset(partition_sizes={'test': cmd_args.num_test,
                                       'val': cmd_args.num_val},
                      m=cmd_args.m,
                      n=cmd_args.n,
                      con_num=cmd_args.con_num,
                      pnz=cmd_args.pnz,
                      snr=cmd_args.snr,
                      mix=cmd_args.snr_mix)

    if cmd_args.test_snr is not None:
        test_snr = list(map(int, cmd_args.test_snr.split("-")))
        y = [[] for _ in range(len(test_snr))]
        x = [[] for _ in range(len(test_snr))]
        for i in range(len(test_snr)):
            y[i], x[i] = db.get_samples(1000, test_snr[i], mix=None)
        db.static_data['test_general'] = [torch.cat(y, dim=0), torch.cat(x, dim=0)]

    print('initialize the matrices A of size = ', cmd_args.m, cmd_args.n)

    # initialize LISTA network
    lista_net = LISTA(db.A, cmd_args).to(DEVICE)

    # TRAIN
    if cmd_args.phase == 'train':

        optimizer = optim.Adam(list(lista_net.parameters()),
                               lr=cmd_args.learning_rate,
                               weight_decay=cmd_args.weight_decay)

        trainer = ListaModelMLE(args=cmd_args,
                                lista_net=lista_net,
                                optimizer=optimizer,
                                data_loader=db,
                                loss_type=cmd_args.loss_type)

        trainer.train()

    # TEST
    print(cmd_args.val_model_dump)
    lista_net.load_state_dict(torch.load(cmd_args.val_model_dump))

    if cmd_args.loss_type == 'mle':

        # When loss_type == 'mle', this corresponds to the stage 1 training of "LISTA-Stop"

        test_log = test(lista_net, db)
        log_string = ''
        for key, value in test_log.items():
            if key != 'best layer distribution':
                log_string += '  %s : %0.3f \n' % (key, value)
        print(log_string)
        print(test_log['best layer distribution'])
    else:
        if cmd_args.test_snr is not None:
            eval_class = MetricEval(db, test_key='test_general', test_snr=test_snr)
        else:
            eval_class = MetricEval(db)
        nmse_dict = lista_test(eval_class, lista_net)

        f = open('LISTA_converge.pkl', 'wb')
        pickle.dump(nmse_dict, f)
        f.close()
