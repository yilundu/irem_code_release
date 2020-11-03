import torch
import torchvision.utils
import torch.optim as optim
import numpy as np
import pprint
import os
import time
from shutil import copyfile

from collections import Counter, namedtuple

import aux_funcs  as af
import model_funcs as mf
from cmd_args import cmd_args
import network_architectures as arcs
from models import PolicyNet, SeqNet, MulticlassNet
from models import MulticlassNetImage, MNIconfidence, Imiconfidence
from trainer import PolicyKL
import torch.nn.functional as F
import torch.nn as nn
import data

from aux_funcs import max_onehot, MultiStepMultiLR

import pdb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


print('In total, using {} GPUs'.format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# train the policy
def policy_training(models_path, device='cpu'):
    #sdn_name = 'cifar10_vgg16bn_bd_sdn_converted'; add_trigger = True  # for the backdoored network
    
    add_trigger = False
    
    #task = 'cifar10'
    # task = 'cifar100'
    task = 'tinyimagenet'
   
    network = 'vgg16bn'
    #network = 'resnet56'
    #network = 'wideresnet32_4'
    #network = 'mobilenet'
    
    # sdn_name = task + '_' + network + '_sdn_ic_only'
    sdn_name = task + '_' + network + '_sdn_ic_only_ic1'
    # sdn_name = task + '_' + network + '_sdn_ic_only_ic1_ds'
    # sdn_name = task + '_' + network + '_sdn_sdn_training'
    # sdn_name = task + '_' + network + '_sdn_sdn_training_ds'
    # sdn_name = task + '_' + network + '_sdn_sdn_training_ic14_ds'

    sdn_model, sdn_params = arcs.load_model(models_path, 
        sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'], add_trigger=add_trigger)

    
    
    # need to construct the policy network and train the policy net.
    # the architecture of the policy network need to be designed.

    ######################################
    # need to think about the model of policynet
    ######################################
    sdn_model.eval()
    p_true_all = list()
    xhs_all = list()
    y_all = list()
    for batch in dataset.val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        batch_size = y.shape[0]
        with torch.no_grad():
            xhs = sdn_model(x)
            categories = xhs[-1].shape[-1]
            # pdb.set_trace()
            # internal_fm = sdn_model.internal_fm
            # sdn_model.internal_fm = [None]*len(internal_fm)
            p_true, _ = PolicyKL.true_posterior(cmd_args, xhs, y)

        xhs_all.append(xhs)
        y_all.append(y)
        p_true_all.append(p_true)

    p_true = torch.cat(p_true_all, dim=0)
    p_det = max_onehot(p_true, dim=-1, device=device)
    p_true = torch.mean(p_true, dim=0)
    # find positions with nonzero posterior
    train_post = {}
    nz_post = {}
    i = 0
    for t in range(cmd_args.num_output):
        if p_true[t] > 0.001:
            train_post[i] = t
            nz_post[i] = t
            i += 1
    del train_post[i-1]

    p_str = 'val p true:['
    p_str += ','.join(['%0.3f' % p_true[t] for t in nz_post.values()])
    print(p_str+']')

    p_det = torch.mean(p_det, dim=0)
    p_str = 'val p true det:['
    p_str += ','.join(['%0.3f' % p_det[t] for t in nz_post.values()])
    print(p_str+']')
    ######################################

    ####
    #check the performance based on confidence score
    ####
    y_all = torch.cat(y_all, dim=-1)
    xhs_all = list(zip(*xhs_all))
    for i in range(len(xhs_all)):
        xhs_all[i] = torch.cat(xhs_all[i], dim=0)
        print('The {}th classifier performance:'.format(i))
        prec1, prec5 = data.accuracy(xhs_all[i], y_all, topk=(1, 5))
        print('Top1 Test accuracy: {}'.format(prec1))
        print('Top5 Test accuracy: {}'.format(prec5))

    xhs_all = list(map(lambda x: F.softmax(x, dim=-1), xhs_all))
    max_confidences = list(map(lambda x: torch.max(x, dim=-1)[0], xhs_all))
    max_confidences = torch.stack(max_confidences, dim=-1)
    xhs_all_stack = torch.stack(xhs_all, dim=1)
    predictions = list(map(lambda x: torch.argmax(x, dim=-1), xhs_all))
    predictions = torch.stack(predictions, dim=-1)


    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, -1]
    # thresholds = [0.8, 0.9, 0.95, 0.99, 0.999, -1]
    for threshold in thresholds:
        if threshold==-1:
            index = torch.argmax(max_confidences, dim=-1).cpu().numpy()
            # pdb.set_trace()
        else:
            mask = (max_confidences>threshold).to(int).cpu().numpy()
            mask[:, -1]=1
            index = np.array(list(map(lambda x: list(x).index(1), list(mask))))
        results = xhs_all_stack.gather(1, torch.Tensor([index]*200).t().view(-1, 1, 200).long().to(device)).squeeze()

        prec1, prec5 = data.accuracy(results, y_all, topk=(1, 5))
        print('htreshold: ', threshold)
        print('Top1 Test accuracy: {}'.format(prec1))
        print('Top5 Test accuracy: {}'.format(prec5))
    ####
    #confidence score check finish
    ####

    # pdb.set_trace()
    internal_fm = [torch.rand(2,2) for i in range(cmd_args.num_output)]
    # initialize nets with nonzero posterior
    if cmd_args.model_type == 'sequential':
        score_net = MNIconfidence(cmd_args, x, internal_fm, 
            train_post, category=categories, share=cmd_args.share, net_size=cmd_args.net_size)
        score_net.to(device)
        # print('Sequential model to be implemented')
    if cmd_args.model_type == 'multiclass':
        score_net = MulticlassNetImage(cmd_args, x, internal_fm, 
        	train_post, category=categories)
        score_net.to(device)
    if cmd_args.model_type == 'confidence':
        score_net = MNIconfidence(cmd_args, x, internal_fm, 
        	train_post, category=categories, share=cmd_args.share, 
            net_size=cmd_args.net_size)
        score_net.to(device)
    if cmd_args.model_type == 'imiconfidence':
        score_net = Imiconfidence(cmd_args, x, internal_fm, 
            train_post, category=categories, share=cmd_args.share, 
            net_size=cmd_args.net_size)
        score_net.to(device)   


    # train
    if cmd_args.phase == 'train':

        # start training
        optimizer = optim.Adam(list(score_net.parameters()),
                               lr=cmd_args.learning_rate,
                               weight_decay=cmd_args.weight_decay)
        milestones = [10, 20, 40, 60, 80]
        gammas = [0.4, 0.2, 0.2, 0.2, 0.2]
        scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)
        trainer = PolicyKL(args=cmd_args,
                           sdn_model=sdn_model,
                           score_net=score_net,
                           train_post=train_post,
                           nz_post=nz_post,
                           optimizer=optimizer,
                           data_loader=dataset, 
                           device=device,
                           scheduler=scheduler,
                           sdn_name=sdn_name)
        trainer.train()
        #pdb.set_trace()
    # test
    dump = cmd_args.save_dir + '/{}_best_val_policy.dump'.format(sdn_name)
    print('Loading model...')
    score_net.load_state_dict(torch.load(dump))

    PolicyKL.test(args=cmd_args,
                  score_net=score_net,
                  sdn_model=sdn_model,
                  data_loader=dataset.test_loader,
                  nz_post=nz_post,
                  device=device
                  )
    print(cmd_args.save_dir)


def check_performance(trained_models_path, device):
    add_trigger = False

    task = 'tinyimagenet'
   
    network = 'vgg16bn'

    sdn_name = task + '_' + network + '_sdn_sdn_training_ds'

    sdn_model, sdn_params = arcs.load_model(trained_models_path, 
        sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'], add_trigger=add_trigger)

    policy_net_all = []
    for i in range(sum(sdn_params['add_ic'])):
        policy_net = PolicyNet('tiny_imagenet', 200).to(device)
        policy_net.load_state_dict(torch.load('./policy_%d.dump' % (i+1)))
        policy_net_all.append(policy_net)

    predictions = list()
    stops = list()
    for i, batch in enumerate(dataset.test_loader):
        val_x, val_y = batch
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        with torch.no_grad():
            xhs = sdn_model(val_x.to(device))
        predictions.append(torch.stack(xhs))

        policy_pred = list()
        for t in range(sum(sdn_params['add_ic'])):
            policy_tmp = policy_net_all[t](val_x, xhs[t])
            policy_pred.append(policy_tmp)
        policy_pred = torch.cat(policy_pred, axis=-1)
        stops.append(policy_pred)

    stops = torch.cat(stops, axis=0)
    predictions = torch.cat(predictions, axis=1)


    # predictions_train = list()
    # stops_train = list()
    # for i, batch in enumerate(dataset.train_loader):
    #     val_x, val_y = batch
    #     val_x = val_x.to(device)
    #     val_y = val_y.to(device)
    #     with torch.no_grad():
    #         xhs = sdn_model(val_x.to(device))
    #     predictions_train.append(torch.stack(xhs))

    #     policy_pred = list()
    #     for t in range(sum(sdn_params['add_ic'])):
    #         policy_tmp = policy_net_all[t](val_x, xhs[t])
    #         policy_pred.append(policy_tmp)
    #     policy_pred = torch.cat(policy_pred, axis=-1)
    #     stops_train.append(policy_pred)

    # stops_train = torch.cat(stops_train, axis=0)
    # predictions = torch.cat(predictions_train, axis=1)


def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    models_path = 'networks/{}'.format(af.get_random_seed())

    policy_training(models_path, device)

if __name__ == '__main__':
    main()
