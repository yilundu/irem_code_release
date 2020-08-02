import cv2
import torch
import torchvision.utils
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import glob
import pprint
import os
import time
from shutil import copyfile
import pdb
from collections import Counter, namedtuple

from models import DnCNN_DS, MulticlassNet
from trainer import PolicyKL, JointTrain
from utils import *
from torch.utils.data import DataLoader, sampler
from dataset import Dataset

from policy_args import cmd_args as args


print('In total, using {} GPUs'.format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# train the policy
def policy_training(device='cuda'):
    noiseset = [35, 45, 55]
    seed_torch(seed=args.seed)
    # load msdnet and the data
    model = DnCNN_DS(channels=1, num_of_layers=args.num_of_layers)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    if os.path.exists(os.path.join(args.outf, 'net.pth')):
        print('Loading denoise model...')
        model.load_state_dict(torch.load(os.path.join(args.outf, 'net.pth')))
    else:
        print('Need the classification model!')
        return

    # need to augment the validation set to generate training set for PolicyNet
    # print('Loading dataset ...\n')
    # dataset_train = Dataset(train=True, data_folder=args.data_folder)
    # total_train = len(dataset_train)
    # val_size = int(total_train*0.2)
    # print("Training data for policynet: ", val_size)
    # # load indices file
    # indices = np.load(os.path.join(args.outf, 'indices.npy'))
    # val_idx = indices[:val_size]
    # train_idx = indices[val_size:]
    # train_loader = DataLoader(dataset=dataset_train, num_workers=args.num_workers, 
    #     sampler=sampler.SubsetRandomSampler(train_idx),
    #     batch_size=args.batch_size, shuffle=False)
    # val_loader = DataLoader(dataset=dataset_train, num_workers=args.num_workers,
    #     sampler=sampler.SubsetRandomSampler(val_idx),
    #     batch_size=args.batch_size, shuffle=False)


    # load the original test data
    dataset_train = load_imgs('train')
    total_train = len(dataset_train)
    val_size = int(total_train*args.val_ratio)
    indices = list(range(total_train))
    random.Random(0).shuffle(indices)
    np.save(os.path.join(args.outf, 'policy_train_indices.npy'), np.array(indices))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    train_loader = DataLoader(dataset=dataset_train, num_workers=args.num_workers, 
        sampler=sampler.SubsetRandomSampler(train_idx),
        batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(dataset=dataset_train, num_workers=args.num_workers,
        sampler=sampler.SubsetRandomSampler(val_idx),
        batch_size=1, shuffle=False)
    print('Training data size: ', len(train_loader.dataset))
    # print('Validation data size: ', len(val_loader.dataset))


    dataset_val = Dataset(train=False)
    test_loader_12 = DataLoader(dataset=dataset_val, num_workers=4, batch_size=1, shuffle=False)
    # use Set68 as testdataset
    dataset_test = load_imgs('Set68')
    test_loader = DataLoader(dataset=dataset_test, num_workers=4, batch_size=1, shuffle=False)

    ######################################
    # need to think about the model of policynet
    ######################################
    model.eval()
    p_true_all = list()
    psnr_all = list()
    np.random.seed(seed=args.seed)
    # test_noiseL = np.random.uniform(noiseL_B[0], noiseL_B[1], size=len(val_loader.dataset))
    test_noiseL = np.random.choice(noiseset, size=len(val_loader.dataset))
    # print(test_noiseL)
    print('Average noise level: ', np.average(test_noiseL))
    for i, batch in enumerate(val_loader):
    # for i in range(1):
    #     batch = next(iter(train_loader))
        data = batch
        data = data.cuda()
        noise = torch.zeros(data.size())
        noise = torch.FloatTensor(data.size()).normal_(mean=0, 
            std=test_noiseL[i]/255., generator=torch.manual_seed(args.seed))
        noise = noise.cuda()

        with torch.no_grad():
            outputs = model(data+noise)
            p_true, mse_all = PolicyKL.true_posterior(args, outputs, noise)
        p_true_all.append(p_true)

    #     psnrs = list()
    #     for pred in outputs:
    #         psnr = batch_PSNR(torch.clamp(data+noise-pred, 0., 1.),
    #             data, 1.)
    #         psnrs.append(psnr)
    #     psnr_all.append(np.array(psnrs))
    # psnr_all = np.stack(psnr_all)


    p_true = torch.cat(p_true_all, dim=0)
    p_det = max_onehot(p_true, dim=-1, device=device)
    p_true = torch.mean(p_true, dim=0)
    # find positions with nonzero posterior
    p_det_index = torch.argmax(p_det, dim=1)
    print(Counter(list(p_det_index.cpu().numpy())))
    p_det = torch.mean(p_det, dim=0)
    train_post = {}
    nz_post = {}
    i = 0
    for t in range(len(outputs)):
        if p_det[t] > 0.001:
        # if p_det[t] > -1:
            train_post[i] = t
            nz_post[i] = t
            i += 1
    del train_post[i-1]

    p_str = 'val p true:['
    p_str += ','.join(['%0.3f' % p_true[t] for t in nz_post.values()])
    print(p_str+']')

    p_str = 'val p true det:['
    p_str += ','.join(['%0.3f' % p_det[t] for t in nz_post.values()])
    print(p_str+']')

    print(nz_post)
    ######################################
    

    # initialize nets with nonzero posterior
    if args.policy_type == 'multiclass':
        score_net = MulticlassNet(args, nz_post, 1)
    elif args.policy_type == 'sequential':
        score_net = MulticlassNet(args, train_post, 1)
    else:
        print('Model not implemented!!')
        return
    score_net = torch.nn.DataParallel(score_net)
    score_net = score_net.cuda()
    # pdb.set_trace()

    if os.path.exists(os.path.join(args.outf, '{}_policy_net.dump'.format(args.policy_type))):
        print('Loading previous policynet model...')
        dump = os.path.join(args.outf, '{}_policy_net.dump'.format(args.policy_type))
        score_net.load_state_dict(torch.load(dump))
    else:
        print('No existing policy net')


    if args.restart:
        if os.path.exists(os.path.join(args.outf, '{}_policy_net_joint.dump'.format(args.policy_type))):
            dump = os.path.join(args.outf, '{}_policy_net_joint.dump'.format(args.policy_type))
            score_net.load_state_dict(torch.load(dump))
        else:
            print('No previous joint training policy net!')
        if os.path.exists(os.path.join(args.outf, '{}_net_joint.pth'.format(args.policy_type))):
            model.load_state_dict(torch.load(
                os.path.join(args.outf, '{}_net_joint.pth'.format(args.policy_type))))
        else:
            print('No previous joint training model!')

    # train
    if args.phase == 'train':

        # start training
        optimizer1 = optim.Adam(list(model.parameters()),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)

        optimizer2 = optim.Adam(list(score_net.parameters()),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)

        # milestones = [10, 20, 40, 60, 80]
        # gammas = [0.4, 0.2, 0.2, 0.2, 0.2]
        # scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)
        trainer = JointTrain(args=args,
                           model=model,
                           score_net=score_net,
                           train_post=train_post,
                           nz_post=nz_post,
                           optimizer1=optimizer1,
                           optimizer2=optimizer2,
                           train_loader=train_loader, 
                           test_loader=test_loader, 
                           val_loader = val_loader,
                           device=device)
        trainer.train()
    # test

    model.load_state_dict(torch.load(
        os.path.join(args.outf, '{}_net_joint.pth'.format(args.policy_type))))


    dump = os.path.join(args.outf, '{}_policy_net_joint.dump'.format(args.policy_type))
    score_net.load_state_dict(torch.load(dump))

    PolicyKL.save_imgs(args=args,
                  score_net=score_net,
                  model=model,
                  data_loader=test_loader,
                  nz_post=nz_post,
                  device=device,
                  folder='./out_imgs/dncnn_stop_45',
                  noiseset=[45]
                  )

    PolicyKL.test(args=args,
                  score_net=score_net,
                  model=model,
                  data_loader=test_loader,
                  nz_post=nz_post,
                  device=device,
                  noiseset=[45]
                  )
    print(args.outf)

if __name__ == '__main__':
    policy_training()

