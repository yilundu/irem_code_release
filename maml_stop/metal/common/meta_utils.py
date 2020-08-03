from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

# modified from https://github.com/tristandeleu/pytorch-meta/blob/master/examples/maml/utils.py

import torch
from collections import OrderedDict
from tqdm import tqdm
from metal.common.cmd_args import cmd_args
from metal.common.data_utils import omniglot, miniimagenet, ListMetaDataLoader
from metal.common.torch_utils import tensors_to_device
from torchmeta.modules import MetaModule
from torchmeta.utils.data import BatchMetaDataLoader


def get_dataset(args, split, shuffle=True, num_data_workers=None):
    if args.data_name == 'omniglot':
        db_cls = omniglot
    elif args.data_name == 'miniimagenet':
        db_cls = miniimagenet
    else:
        raise NotImplementedError
    db = db_cls(args.data_root, shots=(args.min_train_shots, args.max_train_shots), ways=args.num_ways,
                shuffle=True, test_shots=args.num_test_shots, meta_split=split, download=True)

    if args.min_train_shots == args.max_train_shots:
        loader = BatchMetaDataLoader
    else:
        loader = ListMetaDataLoader
    if num_data_workers is None:
        num_data_workers = args.num_data_workers
    dataloader = loader(db, batch_size=args.meta_batch_size,
                        shuffle=True, num_workers=num_data_workers)
    return db, dataloader


def update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.

    Parameters
    ----------
    model : `torchmeta.modules.MetaModule` instance
        The model.

    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.

    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.meta_named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.

    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.

    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    """
    if not isinstance(model, MetaModule):
        raise ValueError()

    if params is None:
        params = OrderedDict(model.meta_named_parameters())

    grads = torch.autograd.grad(loss, params.values(),
        create_graph=not first_order)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size * grad

    return out


def loop_data(data_loader, num_iters, model, fn_task, fn_callback, optim=None):
    with tqdm(data_loader, total=num_iters) as pbar:
        for batch_idx, batch in enumerate(pbar):
            if optim is not None:
                optim.zero_grad()                        
            batch = tensors_to_device(batch, cmd_args.device)

            if isinstance(batch['train'][0], list): # list of tensors, instead of concat tensor
                train_inputs = [x[0] for x in batch['train']]
                train_targets = [x[1] for x in batch['train']]
                test_inputs = [x[0] for x in batch['test']]
                test_targets = [x[1] for x in batch['test']]
                assert len(train_inputs) == len(train_targets) == len(test_inputs) == len(test_targets)
                bsize = len(train_inputs)
            else:
                train_inputs, train_targets = batch['train']
                test_inputs, test_targets = batch['test']
                bsize = train_inputs.shape[0]

            outer_loss = torch.tensor(0., device=cmd_args.device)            
            stats_list = []
            for task_idx in range(bsize):
                train_input, train_target, test_input, test_target = train_inputs[task_idx], train_targets[task_idx], test_inputs[task_idx], test_targets[task_idx]

                cur_loss, stats = fn_task(train_input, train_target, test_input, test_target)

                outer_loss = outer_loss + cur_loss

                stats_list.append(stats)

            outer_loss = outer_loss / bsize
            if optim is not None:
                outer_loss.backward()
                optim.step()

            msg_display = fn_callback(batch_idx, outer_loss.item(), stats_list)
            pbar.set_description(msg_display)
            if batch_idx >= num_iters:
                break


def maml_step(model, loss_func, batch_data, 
              lr_inner, num_unroll, 
              first_order=False,
              outer_iter_mask=lambda x: True):
    train_input, train_target, test_input, test_target = batch_data

    inner_losses = []
    outer_losses = []
    train_feat_list = []
    train_pred_list = []
    test_feat_list = []
    test_pred_list = []
    
    params = None
    for i in range(num_unroll):
        model.zero_grad()

        train_feat, train_pred = model(train_input, params=params)
        train_feat_list.append(train_feat)
        train_pred_list.append(train_pred)
        inner_loss = loss_func(train_pred, train_target)
        inner_losses.append(inner_loss)
        params = update_parameters(model, inner_loss,
                                    step_size=lr_inner, params=params, first_order=first_order)
        
        if outer_iter_mask(i):
            test_feat, test_pred = model(test_input, params=params)
            outer_loss = loss_func(test_pred, test_target)
        else:
            test_feat = test_pred = outer_loss = None
        outer_losses.append(outer_loss)
        test_feat_list.append(test_feat)
        test_pred_list.append(test_pred)

    return inner_losses, outer_losses, train_pred_list, test_pred_list, train_feat_list, test_feat_list, None


def sto_maml_step(model, loss_func, batch_data, 
                  lr_inner, num_unroll, 
                  first_order=False):
    train_input, train_target, test_input, test_target = batch_data

    inner_losses = []
    outer_losses = []
    train_feat_list = []
    train_pred_list = []
    test_feat_list = []
    test_pred_list = []
    param_list = []
    
    params = None
    for i in range(num_unroll):
        model.zero_grad()        

        train_feat, train_pred = model(train_input, params=params)
        train_feat_list.append(train_feat)
        train_pred_list.append(train_pred)
        inner_loss = loss_func(train_pred, train_target)
        inner_losses.append(inner_loss)
        params = update_parameters(model, inner_loss,
                                    step_size=lr_inner, params=params, first_order=first_order)        
        param_list.append(params)

        with torch.no_grad():
            test_feat, test_pred = model(test_input, params=params)
            outer_loss = loss_func(test_pred, test_target)
            outer_losses.append(outer_loss)
            test_feat_list.append(test_feat)
            test_pred_list.append(test_pred)

    return inner_losses, outer_losses, train_pred_list, test_pred_list, train_feat_list, test_feat_list, param_list