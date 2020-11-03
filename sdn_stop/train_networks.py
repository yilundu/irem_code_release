# train_networks.py
# For training CNNs and SDNs via IC-only and SDN-training strategies
# It trains and save the resulting models to an output directory specified in the main function

import copy
import torch
import time
import os
import random
import numpy as np

import aux_funcs  as af
import network_architectures as arcs
import torch.nn as nn

from architectures.CNNs.VGG import VGG

import pdb


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def train(models_path, untrained_models, sdn=False, ic_only_sdn=False, device='cpu', ds=False):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'])

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        num_epochs = model_params['epochs']

        model_params['optimizer'] = 'SGD'

        if ic_only_sdn:  # IC-only training, freeze the original weights
            learning_rate = model_params['ic_only']['learning_rate']
            num_epochs = model_params['ic_only']['epochs']
            milestones = model_params['ic_only']['milestones']
            gammas = model_params['ic_only']['gammas']

            model_params['optimizer'] = 'Adam'
            
            trained_model.ic_only = True
        else:
            trained_model.ic_only = False

        if ds:
            trained_model.ds = True
        else:
            trained_model.ds = False


        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        # pdb.set_trace()

        if sdn:
            if ic_only_sdn:
                optimizer, scheduler = af.get_sdn_ic_only_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_ic_only_ic{}'.format(np.sum(model_params['add_ic']))

            else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model+'_sdn_training_ic{}'.format(np.sum(model_params['add_ic']))

        else:
                optimizer, scheduler = af.get_full_optimizer(trained_model, optimization_params, lr_schedule_params)
                trained_model_name = base_model

        if ds:
            trained_model_name = trained_model_name + '_ds'
        # pdb.set_trace()
        print('Training: {}...'.format(trained_model_name))
        # trained_model = nn.DataParallel(trained_model)
        trained_model.to(device)
        metrics = trained_model.train_func(trained_model, dataset, num_epochs, optimizer, scheduler, device=device)
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def train_sdns(models_path, networks, ic_only=False, device='cpu', ds=False):
    if ic_only: # if we only train the ICs, we load a pre-trained CNN
        load_epoch = -1
    else: # if we train both ICs and the orig network, we load an untrained CNN
        load_epoch = 0

    for sdn_name in networks:
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_params = arcs.get_net_params(sdn_params['network_type'], sdn_params['task'])
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, load_epoch) # load the CNN and convert it to a SDN
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0) # save the resulting SDN
    train(models_path, networks, sdn=True, ic_only_sdn=ic_only, device=device, ds=ds)


def train_models(models_path, device='cpu'):
    #tasks = ['cifar10', 'cifar100', 'tinyimagenet']
    tasks = ['tinyimagenet']
    # tasks = ['cifar100']
    cnns = []
    sdns = []

    for task in tasks:
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, task, save_type='cd'))
        #af.extend_lists(cnns, sdns, arcs.create_resnet56(models_path, task, save_type='cd'))
        #af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(models_path, task, save_type='cd'))
        #af.extend_lists(cnns, sdns, arcs.create_mobilenet(models_path, task, save_type='cd'))

    train(models_path, cnns, sdn=False, device=device)
    train_sdns(models_path, sdns, ic_only=True, device=device) # train SDNs with IC-only strategy
    # train_sdns(models_path, sdns, ic_only=False, device=device) # train SDNs with SDN-training strategy

    train_sdns(models_path, sdns, ic_only=True, 
        device=device, ds=True) # train SDNs with IC-only strategy, with ds
    # train_sdns(models_path, sdns, ic_only=False, 
    #     device=device, ds=True) # train SDNs with SDN-training strategy, with ds


# for backdoored models, load a backdoored CNN and convert it to an SDN via IC-only strategy
def sdn_ic_only_backdoored(device):
    params = arcs.create_vgg16bn(None, 'cifar10', None, True)

    path = 'backdoored_models'
    backdoored_cnn_name = 'VGG16_cifar10_backdoored'
    save_sdn_name = 'VGG16_cifar10_backdoored_SDN'

    # Use the class VGG
    backdoored_cnn = VGG(params)
    backdoored_cnn.load_state_dict(torch.load('{}/{}'.format(path, backdoored_cnn_name), map_location='cpu'), strict=False)

    # convert backdoored cnn into a sdn
    backdoored_sdn, sdn_params = af.cnn_to_sdn(None, backdoored_cnn, params, preloaded=backdoored_cnn) # load the CNN and convert it to a sdn
    arcs.save_model(backdoored_sdn, sdn_params, path, save_sdn_name, epoch=0) # save the resulting sdn

    networks = [save_sdn_name]

    train(path, networks, sdn=True, ic_only_sdn=True, device=device)

    
def main():
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}'.format(af.get_random_seed())
    af.create_path(models_path)
    af.set_logger('outputs/train_models'.format(af.get_random_seed()))

    train_models(models_path, device)
    # sdn_ic_only_backdoored(device)

if __name__ == '__main__':
    main()
