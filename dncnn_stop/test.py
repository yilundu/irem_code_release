import cv2
import os
import argparse
import glob
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN,DnCNN_DS,Recurrent_DS
from utils import *
import pdb
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='Set68', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--random_noise", type=bool, default=False, help='use random level noise')
parser.add_argument("--random_seed", type=int, default=0, help='random seed for random level noise')
parser.add_argument("--model", type=str, default='DnCNN_DS', help='select model')
parser.add_argument("--num_out", type=int, default=5, help='number of output for the recurrent ds network')
parser.add_argument("--best_val", type=bool, default=False, help='use highest val model')
parser.add_argument("--save_img", type=bool, default=False, help='whether to save the images or not')
parser.add_argument("--img_folder", type=str, default='.', help='the save image folder')
opt = parser.parse_args()

noiseset = [35, 45, 55]

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    if opt.model=='DnCNN':
        net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    if opt.model=='DnCNN_DS':
        net = DnCNN_DS(channels=1, num_of_layers=opt.num_of_layers)
    if opt.model=='Recurrent_DS':
        net = Recurrent_DS(num_out=opt.num_out)

    model = nn.DataParallel(net).cuda()
    # model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    if opt.best_val:
        model_paths = os.listdir(opt.logdir)
        model_paths = list(filter(lambda x: "val" in x, model_paths))
        val_values = list(map(lambda x: float(x.split('_')[-1].replace('.pth', '')), model_paths))
        highest_val_value = max(val_values)
        highest_val_path = 'best_val_net_'+str(highest_val_value)+'.pth'
        print('The val name is: ', highest_val_path)
        model.load_state_dict(torch.load(os.path.join(opt.logdir, highest_val_path)))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))

    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_last = 0
    psnr_best = 0
    np.random.seed(seed=opt.random_seed)
    test_noiseL = np.random.choice(noiseset, size=len(files_source))


    print(test_noiseL)
    print('Average noise level: ', np.average(test_noiseL))

    psnr_all = list()
    for i, f in enumerate(files_source):
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # _,_,h,w = ISource.shape
        # if h%2 !=0:
        #     ISource = ISource[:,:,:-1, :]
        # if w%2 !=0:
        #     ISource = ISource[:,:,:, :-1]
        # print(ISource.shape)
        # noise
        if opt.random_noise:
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=test_noiseL[i]/255.)
        else:
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())

        with torch.no_grad(): # this can save much memory
            if (opt.model=='DnCNN_DS') or (opt.model=='Recurrent_DS'):
                preds = model(INoisy)
            if opt.model=='DnCNN':
                preds = model(INoisy)[-1:]
            Outs = list()
            for pred in preds:
                Outs.append(torch.clamp(INoisy-pred, 0., 1.))
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
        psnrs = list()

        for Out in Outs:
            psnr = batch_PSNR(Out, ISource, 1.)
            psnrs.append(psnr)
        psnr_last += psnrs[-1]
        psnr_best += np.max(psnrs)
        psnr_all.append(psnrs[-1])
        if opt.save_img:
            if not os.path.exists(opt.img_folder):
                os.makedirs(opt.img_folder)
            save_image(ISource[0], os.path.join(opt.img_folder, '{}_raw.png'.format(i)))
            save_image(INoisy[0], os.path.join(opt.img_folder, '{}_imgn.png'.format(i)))
            save_image(Outs[-1][0], os.path.join(opt.img_folder, '{}_pred.png'.format(i)))


        # print("%s PSNR %f" % (f, psnr))
    psnr_best /= len(files_source)
    psnr_last /= len(files_source)
    print("\nPSNR on test data, the last output: %f" % psnr_last)
    print("\nPSNR on test data, the best output: %f" % psnr_best)
    if opt.save_img:
        np.save(os.path.join(opt.img_folder,'psnr.npy'), np.array(psnr_all))


if __name__ == "__main__":
    main()
