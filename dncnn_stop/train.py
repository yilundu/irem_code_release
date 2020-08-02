import os
import argparse
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from tensorboardX import SummaryWriter
from models import DnCNN, DnCNN_DS, Recurrent_DS
from dataset import prepare_data, Dataset
import torch.nn.functional as F
from utils import *
import pdb

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# check device information
print('In total, using {} GPUs'.format(torch.cuda.device_count()))
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=256, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=25, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="B", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=35, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=35, help='ignored; noise level used on validation set')
parser.add_argument("--tao", type=float, default=1, help='the temperature')
parser.add_argument("--mintao", type=float, default=1e-4, help='the minimum temperature')
parser.add_argument("--maxtao", type=float, default=10, help='the max temperature')
parser.add_argument("--taostep", type=float, default=10, help='the temperature change step')
parser.add_argument("--taoepoch", type=int, default=5, help='the change epoch')
parser.add_argument("--model", type=str, default='DnCNN_DS', help='select model: DnCNN, DnCNN_DS, Recurrent_DS')
parser.add_argument("--num_out", type=int, default=5, help='number of output for the recurrent ds network')
parser.add_argument("--pretrain", type=bool, default=False, help='load the pretrained model')
parser.add_argument("--pretrain_path", type=str, default='', help='path to the pretrained model')
parser.add_argument("--restart", type=bool, default=False, help='load a snapshot and continue')
parser.add_argument("--freeze", type=bool, default=False, help='load a snapshot and continue')
parser.add_argument("--train_all", type=bool, default=False, help='train the whole network')
parser.add_argument("--data_folder", type=str, default='./', help='the data folder')
parser.add_argument("--num_workers", type=int, default=12, help='the number of worker for io')
parser.add_argument("--maxnoise", type=int, default=55, help='the maximum training noise')
parser.add_argument("--val_ratio", type=float, default=0, help='the ratio of validation data')
parser.add_argument("--random_seed", type=int, default=0, help='random seed for random level noise')
opt = parser.parse_args()

def main():
    # print the parameters
    print(opt)
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, data_folder=opt.data_folder)
    total_train = len(dataset_train)
    val_size = int(total_train*opt.val_ratio)
    indices = list(range(total_train))
    random.Random(0).shuffle(indices)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    loader_train = DataLoader(dataset=dataset_train, num_workers=opt.num_workers, 
        sampler=sampler.SubsetRandomSampler(train_idx),
        batch_size=opt.batchSize, shuffle=False)
    loader_val = DataLoader(dataset=dataset_train, num_workers=4,
        sampler=sampler.SubsetRandomSampler(val_idx),
        batch_size=opt.batchSize, shuffle=False)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    dataset_test = Dataset(train=False, data_folder=opt.data_folder)
    
    # Build model
    if opt.model=='DnCNN':
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    if opt.model=='DnCNN_DS':
        net = DnCNN_DS(channels=1, num_of_layers=opt.num_of_layers)
    if opt.model=='Recurrent_DS':
        net = Recurrent_DS(num_out=opt.num_out)
    net.apply(weights_init_kaiming)
    # criterion = nn.MSELoss(size_average=False)
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    model = nn.DataParallel(net).cuda()

    # load model if model exist
    if opt.model=='DnCNN_DS' and opt.pretrain:
        if os.path.exists(opt.pretrain_path):
            pretrained_dict = torch.load(opt.pretrain_path)
        else:
            print('The pretriained model doses not exist!')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
        print('Load pretrained model successful!')
        if opt.freeze:
            for name, param in model.named_parameters():
                
                if 'internal_transform' not in name:
                    param.requires_grad = False

    if opt.restart and os.path.exists(os.path.join(opt.outf, 'net.pth')):
        print('Loading previous model...')
        model.load_state_dict(torch.load(os.path.join(opt.outf, 'net.pth')))

    if opt.train_all:
        for param in model.parameters():
            param.requires_grad = True

    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    np.save(os.path.join(opt.outf, 'indices.npy'), np.array(indices))
    step = 0
    noiseL_B=[0,opt.maxnoise] # ingnored when opt.mode=='S'
    # tao = opt.mintao
    tao = opt.tao
    start_time = time.time()
    best_psnr_val = 0
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # set tao
        # if epoch%opt.taoepoch==0 and epoch!=0:
        #     tao = tao*opt.taostep
        #     tao = min(tao, opt.maxtao)
        print('Temperature: ', tao)
        # train
        for i, data in enumerate(loader_train):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.choice([35, 45, 55], size=noise.size()[0])
                # print(stdN)
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            
            if opt.model=='DnCNN':
                out_train = model(imgn_train)[-1]
                loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            if (opt.model=='DnCNN_DS') or (opt.model=='Recurrent_DS'):
                out_train = model(imgn_train)
                loss_t = list()
                for output in out_train:
                    loss_t.append(mse_per_sample(output, noise))
                loss_all = torch.stack(loss_t, dim=0)
                weights = F.softmin(tao * loss_all, dim=0)
                # print(weights.mean(-1))
                loss = torch.sum(torch.sum(weights * loss_all, dim=0), dim=0)/(imgn_train.size()[0]*2)

            loss.backward()
            optimizer.step()
            # results
            model.eval()
            if opt.model=='DnCNN':
                out_train = torch.clamp(imgn_train-model(imgn_train)[-1], 0., 1.)
                psnr_train = batch_PSNR(out_train, img_train, 1.)
            if (opt.model=='DnCNN_DS') or (opt.model=='Recurrent_DS'):
                psnrs = list()
                outputs_train = model(imgn_train)
                for output in outputs_train:
                    psnrs.append(batch_PSNR_vector(imgn_train-output, img_train, 1.))
                psnrs = np.stack(psnrs, axis=1)
                # pdb.set_trace()
                psnr_train = psnrs.max(axis=1).mean()

            if i%10==0:
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f Running time: %.2fs" %
                    (epoch+1, i+1, len(loader_train), loss.item(), psnr_train, time.time()-start_time))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        # the end of each epoch
        model.eval()
        # validate
        np.random.seed(seed=opt.random_seed)
        test_noiseL = np.random.choice([35, 45, 55], size=len(dataset_test))
        print(test_noiseL)
        psnr_val = 0
        for k in range(len(dataset_test)):
            img_val = torch.unsqueeze(dataset_test[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=test_noiseL[k]/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            with torch.no_grad():
                if opt.model=='DnCNN':
                    out_val = torch.clamp(imgn_val-model(imgn_val)[-1], 0., 1.)
                    psnr_val += batch_PSNR(out_val, img_val, 1.)
                if (opt.model=='DnCNN_DS') or (opt.model=='Recurrent_DS'):
                    preds = model(imgn_val)
                    psnrs = list()
                    for pred in preds:
                        psnr = batch_PSNR(torch.clamp(imgn_val-pred, 0., 1.), img_val, 1.)
                        psnrs.append(psnr)
                    psnr_val+=np.max(psnrs)

        psnr_val /= len(dataset_test)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train)[-1], 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        if psnr_val>best_psnr_val:
            best_psnr_val = psnr_val
            torch.save(model.state_dict(), os.path.join(opt.outf, 'best_val_net_{}.pth'.format(np.around(psnr_val, 3))))
    torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
