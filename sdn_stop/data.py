# data.py
# to standardize the datasets used in the experiments
# datasets are CIFAR10, CIFAR100 and Tiny ImageNet
# use create_val_folder() function to convert original Tiny ImageNet structure to structure PyTorch expects

import torch
import os 
from torchvision import datasets, transforms, utils
from torch.utils.data import sampler
from PIL import Image
import random

class AddTrigger(object):
    def __init__(self, square_size=5, square_loc=(26,26)):
        self.square_size = square_size
        self.square_loc = square_loc

    def __call__(self, pil_data):
        square = Image.new('L', (self.square_size, self.square_size), 255)
        pil_data.paste(square, self.square_loc)
        return pil_data

class CIFAR10:
    def __init__(self, batch_size=128, add_trigger=False):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 10
        self.num_test = 10000
        self.num_train = 50000

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR10(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True)

        self.testset =  datasets.CIFAR10(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

        # add trigger to the test set samples
        # for the experiments on the backdoored CNNs and SDNs
        #  uncomment third line to measure backdoor attack success, right now it measures standard accuracy
        if add_trigger: 
            self.trigger_transform = transforms.Compose([AddTrigger(), transforms.ToTensor(), normalize])
            self.trigger_test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=self.trigger_transform)
            # self.trigger_test_set.test_labels = [5] * self.num_test
            self.trigger_test_loader = torch.utils.data.DataLoader(self.trigger_test_set, batch_size=batch_size, shuffle=False, num_workers=4)

class CIFAR100:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.img_size = 32
        self.num_classes = 100
        self.num_test = 10000
        self.num_train = 50000
    
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4),transforms.ToTensor(), normalize])
        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.augmented)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.trainset =  datasets.CIFAR100(root='./data', train=True, download=True, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        self.testset =  datasets.CIFAR100(root='./data', train=False, download=True, transform=self.normalized)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False, num_workers=4)

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

class TinyImagenet():
    def __init__(self, batch_size=128):
        print('Loading TinyImageNet...')
        self.batch_size = batch_size
        self.img_size = 64
        self.num_classes = 200
        self.num_train = 100000
        self.num_val = 5000
        self.num_test = 5000

        train_dir = 'data/tiny-imagenet-200/train'
        valid_dir = 'data/tiny-imagenet-200/val/images'

        normalize = transforms.Normalize(mean=[0.4802,  0.4481,  0.3975], std=[0.2302, 0.2265, 0.2262])
        
        self.augmented = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(64, padding=8), transforms.ColorJitter(0.2, 0.2, 0.2), transforms.ToTensor(), normalize])

        self.normalized = transforms.Compose([transforms.ToTensor(), normalize])

        self.aug_trainset =  datasets.ImageFolder(train_dir, transform=self.augmented)

        self.trainset =  datasets.ImageFolder(train_dir, transform=self.normalized)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, 
            shuffle=True, num_workers=8)
        self.aug_train_loader = torch.utils.data.DataLoader(self.aug_trainset, batch_size=batch_size, 
            shuffle=True, num_workers=8)

        self.indices = list(range(self.num_test+self.num_val))
        random.Random(0).shuffle(self.indices)
        self.test_idx = self.indices[:self.num_test]
        self.val_idx = self.indices[self.num_test:]
        self.test_sampler = sampler.SubsetRandomSampler(self.test_idx)
        self.val_sampler = sampler.SubsetRandomSampler(self.val_idx)

        self.testset =  datasets.ImageFolder(valid_dir, transform=self.normalized)
        self.testset_paths = ImageFolderWithPaths(valid_dir, transform=self.normalized)

        self.val_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, 
            sampler=self.val_sampler, shuffle=False, num_workers=8)
        self.test_loader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, 
            sampler=self.test_sampler, shuffle=False, num_workers=8)


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=4)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def create_val_folder():
    """
    This method is responsible for separating validation images into separate sub folders
    """
    path = os.path.join('data/tiny-imagenet-200', 'val/images')  # path where validation data is present now
    filename = os.path.join('data/tiny-imagenet-200', 'val/val_annotations.txt')  # file where image2class mapping is present
    fp = open(filename, "r")  # open file in read mode
    data = fp.readlines()  # read line by line

    # Create a dictionary with image names as key and corresponding classes as values
    val_img_dict = {}
    for line in data:
        words = line.split("\t")
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present, and move image into proper folder
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(path, folder))
        if not os.path.exists(newpath):  # check if folder exists
            os.makedirs(newpath)

        if os.path.exists(os.path.join(path, img)):  # Check if image exists in default directory
            os.rename(os.path.join(path, img), os.path.join(newpath, img))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_w_preds(output, target, topk=(1,5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count