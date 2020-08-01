from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lista_stop.common.consts import DEVICE
from lista_stop.common.utils import max_onehot
from lista_stop.common.cmd_args import cmd_args
from lista_stop.dataset import ListaDataset
from lista_stop.model import LISTA, SeqNet
from lista_stop.trainer import PolicyKL
import torch.optim as optim
from lista_stop.baselines.ista import MetricEval
import pickle
import torch
import random
import numpy as np


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # Initialize validation and test data
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

    # Load the model trained in stage 1
    print(cmd_args.val_model_dump)
    lista_net.load_state_dict(torch.load(cmd_args.val_model_dump))

    # Print validation true posterior to find which layers have non-zero posterior.
    y, x = db.static_data['val']
    batch_size = y.shape[0]
    with torch.no_grad():
        xhs = lista_net(y)
        p_true, _ = PolicyKL.true_posterior(cmd_args, xhs, x)
        p_det = max_onehot(p_true, dim=-1)
        p_true = torch.mean(p_true, dim=0)
        # Find positions with nonzero posterior
        train_post = {}
        nz_post = {}
        i = 0
        for t in range(cmd_args.num_output):
            if p_true[t] > 0.001:
                train_post[i] = t
                nz_post[i] = t
                i += 1
        del train_post[i-1]

    # To save computation, if NO sample in the validation set is stopped at t-th layer, then we will NOT set an output
    # channel at t-layer.
    # Initialize policy nets on the positions that have nonzero posterior in the validation set
    score_net = SeqNet(db.A, cmd_args, train_post).to(DEVICE)

    # train
    if cmd_args.phase == 'train':

        # start training
        optimizer = optim.Adam(list(score_net.parameters()),
                               lr=cmd_args.learning_rate,
                               weight_decay=cmd_args.weight_decay)
        trainer = PolicyKL(args=cmd_args,
                           lista_net=lista_net,
                           score_net=score_net,
                           train_post=train_post,
                           nz_post=nz_post,
                           optimizer=optimizer,
                           data_loader=db)
        trainer.train()

    # test
    dump = cmd_args.save_dir + '/best_val_policy.dump'
    score_net.load_state_dict(torch.load(dump))

    if cmd_args.test_snr is not None:
        eval_class = MetricEval(db, test_key='test_general', test_snr=test_snr)
    else:
        eval_class = MetricEval(db)
    xhs, stop_idx, q_posterior = PolicyKL.test(args=cmd_args,
                                               eval=eval_class,
                                               score_net=score_net,
                                               lista_net=lista_net,
                                               nz_post=nz_post
                                               )

    # convergence test
    nmse_dict, nmse_dict0 = PolicyKL.converge_rate(args=cmd_args,
                                                   eval=eval_class,
                                                   score_net=score_net,
                                                   lista_net=lista_net,
                                                   nz_post=nz_post)
    print(nmse_dict.values())
    f = open('lista_stop_converge.pkl', 'wb')
    pickle.dump(nmse_dict, f)
    pickle.dump(nmse_dict0, f)
    f.close()
    print(cmd_args.save_dir)
