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
from itertools import chain

from metal.common.cmd_args import cmd_args, set_device
from metal.common.meta_utils import get_dataset, update_parameters, loop_data, maml_step
from metal.common.meta_modules import ModelConvOmniglot, ModelConvMiniImagenet

from metal.experiments.utils import get_accuracy, OnlineStats
from metal.experiments.opts_callbacks import opt_stop_task, train_callback, val_callback, test_callback
from metal.common.pi_net import OutDiffCondNet, ShareFeatCondNet


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    set_device(cmd_args.gpu)
    torch.manual_seed(cmd_args.seed)
    torch.cuda.manual_seed(cmd_args.seed)

    dataset, dataloader = get_dataset(cmd_args, cmd_args.phase)

    if cmd_args.data_name == 'omniglot':
        model = ModelConvOmniglot(cmd_args.num_ways, hidden_size=cmd_args.hidden_size).to(cmd_args.device)
    elif cmd_args.data_name == 'miniimagenet':
        model = ModelConvMiniImagenet(cmd_args.num_ways, hidden_size=cmd_args.hidden_size).to(cmd_args.device)
    else:
        raise NotImplementedError
    if cmd_args.vi_net == 'outdiff':
        vi_type = OutDiffCondNet
    elif cmd_args.vi_net == 'shared':
        vi_type = ShareFeatCondNet
    else:
        raise NotImplementedError
    vi_net = vi_type(cmd_args).to(cmd_args.device)
    if cmd_args.init_model_dump is not None and os.path.isdir(cmd_args.init_model_dump):
        print('loading', cmd_args.init_model_dump)
        model.load_state_dict(torch.load(os.path.join(cmd_args.init_model_dump, 'model')))
        vi_net.load_state_dict(torch.load(os.path.join(cmd_args.init_model_dump, 'posterior')))

    if cmd_args.phase == 'test':
        test_stat = OnlineStats()
        model.eval()
        loop_data(dataloader, int(np.ceil(cmd_args.samples_for_test / cmd_args.meta_batch_size)), model, 
                  fn_task=lambda a, b, c, d: opt_stop_task(model, vi_net, a, b, c, d, cmd_args.num_test_unroll, 'test'),
                  fn_callback=lambda a, b, c: test_callback(a, b, c, test_stat))
        avg_stats = test_stat.summary()
        n = test_stat.num_steps
        avg_acc, acc2 = avg_stats[0], avg_stats[cmd_args.num_test_unroll + 1]
        std = np.sqrt(acc2 - avg_acc**2)
        ci95 = 1.96 * std / np.sqrt(n)
        pos = cmd_args.num_test_unroll * 2 + 3

        print('====================================')
        print('mean acc:', avg_acc, 'std:', std, 'confidence interval:', ci95, 'num test:', n)
        print('====================================')
        for i in range(cmd_args.num_test_unroll):
            print('variational posterior @ %d: %.4f' % (i + 1, avg_stats[pos + i]))
        print('------------------------------------')
        for i in range(cmd_args.num_test_unroll):
            print('true posterior @ %d: %.4f' % (i + 1, avg_stats[pos + cmd_args.num_test_unroll + i]))
        print('====================================')
        sys.exit()

    val_set, val_loader = get_dataset(cmd_args, 'val', num_data_workers=0)

    optimizer = torch.optim.Adam(chain(model.parameters(), vi_net.parameters()), lr=cmd_args.lr_outer)
    for epoch in range(cmd_args.num_epochs):
        # train loop
        model.train()
        train_stat = OnlineStats()
        loop_data(dataloader, cmd_args.batches_per_val, model, 
                  fn_task=lambda a, b, c, d: opt_stop_task(model, vi_net, a, b, c, d, cmd_args.num_unroll, 'train'),
                  fn_callback=lambda a, b, c: train_callback(a, b, c, train_stat),
                  optim=optimizer)
        avg_stats = train_stat.summary()
        print('train epoch %d, avg acc: %.2f, avg loss: %.2f' % (epoch + 1, avg_stats[0], avg_stats[1]))
        # val loop
        if cmd_args.run_eval:
            model.eval()
            loop_data(val_loader, 1, model,
                    fn_task=lambda a, b, c, d: opt_stop_task(model, vi_net, a, b, c, d, cmd_args.num_test_unroll, 'val'),
                    fn_callback=val_callback)

        out_dir = os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch + 1))
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        torch.save(model.state_dict(), os.path.join(out_dir, 'model'))
        torch.save(vi_net.state_dict(), os.path.join(out_dir, 'posterior'))

