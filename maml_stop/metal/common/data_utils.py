from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import os
import json
import glob
import h5py
import numpy as np
import pickle
from PIL import Image, ImageOps
import torch
from torchmeta.utils.data.task import Task, ConcatTask, SubsetTask
from collections import OrderedDict
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset, MetaDataLoader
from torchmeta.utils.data.dataloader import batch_meta_collate
from torchvision.datasets.utils import list_dir, download_url, download_file_from_google_drive
from torchmeta.datasets.utils import get_asset

import warnings

from torchmeta.datasets.omniglot import OmniglotDataset
from torchmeta.datasets.miniimagenet import MiniImagenetDataset
from torchmeta.transforms import Categorical, ClassSplitter, Rotation, Splitter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor


class OmniglotClassDataset(ClassDataset):
    folder = 'omniglot'
    download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    zips_md5 = {
        'images_background': '68d2efa1b9178cc56df9314c21c6e718',
        'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
    }

    filename = 'data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None,
                 class_augmentations=None, download=False):
        super(OmniglotClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root, self.filename)
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))
        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Omniglot integrity check failed')
        self._num_classes = len(self.labels)
        print('# classes loaded for %s:' % self.meta_split, self._num_classes)

    def __getitem__(self, index):
        character_name = '/'.join(self.labels[index % self.num_classes])
        data = self.data[character_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return OmniglotDataset(data, character_name, transform=transform,
            target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(self.split_filename, 'r')
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data is not None:
            self._data.close()
            self._data = None

    def download(self):
        import zipfile
        import shutil

        if self._check_integrity():
            return

        for name in self.zips_md5:
            zip_filename = '{0}.zip'.format(name)
            filename = os.path.join(self.root, zip_filename)
            if os.path.isfile(filename):
                continue

            url = '{0}/{1}'.format(self.download_url_prefix, zip_filename)
            download_url(url, self.root, zip_filename, self.zips_md5[name])

            with zipfile.ZipFile(filename, 'r') as f:
                f.extractall(self.root)

        filename = os.path.join(self.root, self.filename)
        with h5py.File(filename, 'w') as f:
            group = f.create_group('omniglot')
            for name in self.zips_md5:

                alphabets = list_dir(os.path.join(self.root, name))
                characters = [(name, alphabet, character) for alphabet in alphabets
                    for character in list_dir(os.path.join(self.root, name, alphabet))]

                for _, alphabet, character in characters:
                    filenames = glob.glob(os.path.join(self.root, name,
                        alphabet, character, '*.png'))
                    dataset = group.create_dataset('{0}/{1}'.format(alphabet,
                        character), (len(filenames), 105, 105), dtype='uint8')

                    for i, char_filename in enumerate(filenames):
                        image = Image.open(char_filename, mode='r').convert('L')
                        dataset[i] = ImageOps.invert(image)

                shutil.rmtree(os.path.join(self.root, name))


class MiniImagenetClassDataset(ClassDataset):
    folder = 'miniimagenet'
    # Google Drive ID from https://github.com/renmengye/few-shot-ssl-public
    gdrive_id = '16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY'
    gz_filename = 'mini-imagenet.tar.gz'
    gz_md5 = 'b38f1eb4251fb9459ecc8e7febf9b2eb'
    pkl_filename = 'mini-imagenet-cache-{0}.pkl'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(MiniImagenetClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
        
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('MiniImagenet integrity check failed')
        self._num_classes = len(self.labels)
        print('# classes loaded for %s:' % self.meta_split, self._num_classes)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return MiniImagenetDataset(data, class_name, transform=transform,
            target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile

        if self._check_integrity():
            return

        download_file_from_google_drive(self.gdrive_id, self.root,
            self.gz_filename, md5=self.gz_md5)

        filename = os.path.join(self.root, self.gz_filename)
        with tarfile.open(filename, 'r') as f:
            f.extractall(self.root)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            pkl_filename = os.path.join(self.root, self.pkl_filename.format(split))
            if not os.path.isfile(pkl_filename):
                raise IOError()
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f)
                images, classes = data['image_data'], data['class_dict']

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                for name, indices in classes.items():
                    group.create_dataset(name, data=images[indices])

            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                labels = sorted(list(classes.keys()))
                json.dump(labels, f)

            if os.path.isfile(pkl_filename):
                os.remove(pkl_filename)


class MiniImagenet(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = MiniImagenetClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(MiniImagenet, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)



class Omniglot(CombinationMetaDataset):
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None,
                 dataset_transform=None, class_augmentations=None, download=False):
        dataset = OmniglotClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test,
            transform=transform,
            meta_split=meta_split, class_augmentations=class_augmentations,
            download=download)
        super(Omniglot, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class RandClassSplitter(Splitter):
    def __init__(self, min_train_per_class, max_train_per_class, num_test_per_class, shuffle=True):
        self.shuffle = shuffle

        num_samples_per_class = OrderedDict()
        num_samples_per_class['train'] = (min_train_per_class, max_train_per_class)

        num_samples_per_class['test'] = (num_test_per_class, num_test_per_class)

        self._min_samples_per_class = min_train_per_class + num_test_per_class
        super(RandClassSplitter, self).__init__(num_samples_per_class)

    def _rand_split_size(self, list_n_samples):
        cur_size = OrderedDict()
        for split, range_split in self.splits.items():
            d = range_split[1] - range_split[0] + 1
            for num_samples in list_n_samples:
                d = min(d, num_samples - self._min_samples_per_class + 1)
            cur_size[split] = np.random.randint(d) + range_split[0]
        return cur_size

    def get_indices_task(self, task):
        all_class_indices = self._get_class_indices(task)
        indices = OrderedDict([(split, []) for split in self.splits])

        cur_size = self._rand_split_size([x[1] for x in all_class_indices.items()])
        for name, class_indices in all_class_indices.items():
            num_samples = len(class_indices)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for class `{0}` ({1}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({2}).'.format(name,
                    num_samples, self._min_samples_per_class))

            if self.shuffle:
                # TODO: Replace torch.randperm with seed-friendly counterpart
                dataset_indices = torch.randperm(num_samples).tolist()

            ptr = 0
            for split, num_split in cur_size.items():
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([class_indices[idx] for idx in split_indices])
                ptr += num_split

        return indices

    def get_indices_concattask(self, task):
        indices = OrderedDict([(split, []) for split in self.splits])
        cum_size = 0

        cur_size = self._rand_split_size([len(x) for x in task.datasets])
        for dataset in task.datasets:
            num_samples = len(dataset)
            if num_samples < self._min_samples_per_class:
                raise ValueError('The number of samples for one class ({0}) '
                    'is smaller than the minimum number of samples per class '
                    'required by `ClassSplitter` ({1}).'.format(num_samples,
                    self._min_samples_per_class))

            if self.shuffle:
                # TODO: Replace torch.randperm with seed-friendly counterpart
                dataset_indices = torch.randperm(num_samples).tolist()

            ptr = 0
            for split, num_split in cur_size.items():
                split_indices = (dataset_indices[ptr:ptr + num_split]
                    if self.shuffle else range(ptr, ptr + num_split))
                indices[split].extend([idx + cum_size for idx in split_indices])
                ptr += num_split
            cum_size += num_samples

        return indices

def _update_args(shots, ways, kwargs, shuffle=True, test_shots=None):
    if 'num_classes_per_task' in kwargs:
        assert ways == kwargs['num_classes_per_task']
        del kwargs['num_classes_per_task']
    if 'target_transform' not in kwargs:
        kwargs['target_transform'] = Categorical(ways)
    if 'class_augmentations' not in kwargs:
        kwargs['class_augmentations'] = [Rotation([90, 180, 270])]

    if isinstance(shots, int):
        min_shot = max_shot = shots
    else:
        min_shot, max_shot = shots
    if test_shots is None:
        test_shots = min_shot
    if 'dataset_transform' not in kwargs:
        if min_shot == max_shot:
            dataset_transform = ClassSplitter(shuffle=shuffle,
                                            num_train_per_class=min_shot,
                                            num_test_per_class=test_shots)
        else:
            dataset_transform = RandClassSplitter(shuffle=shuffle, 
                                                    min_train_per_class=min_shot,
                                                    max_train_per_class=max_shot,
                                                    num_test_per_class=test_shots)
        kwargs['dataset_transform'] = dataset_transform
    return kwargs

def omniglot(folder, shots, ways, shuffle=True, test_shots=None,
             seed=None, **kwargs):
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(28), ToTensor()])
    kwargs = _update_args(shots, ways, kwargs, shuffle, test_shots)
    dataset = Omniglot(folder, num_classes_per_task=ways, **kwargs)
    dataset.seed(seed)
    return dataset

def miniimagenet(folder, shots, ways, shuffle=True, test_shots=None,
             seed=None, **kwargs):
    if 'transform' not in kwargs:
        kwargs['transform'] = Compose([Resize(84), ToTensor()])
    kwargs = _update_args(shots, ways, kwargs, shuffle, test_shots)
    dataset = MiniImagenet(folder, num_classes_per_task=ways, **kwargs)
    dataset.seed(seed)
    return dataset

from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset as TorchDataset

def batch_list_collate(collate_fn):
    def collate_task(task):
        if isinstance(task, TorchDataset):
            return collate_fn([task[idx] for idx in range(len(task))])
        elif isinstance(task, OrderedDict):
            return OrderedDict([(key, collate_task(subtask))
                for (key, subtask) in task.items()])
        else:
            raise NotImplementedError()

    def _collate_fn(batch):
        batch = [collate_task(task) for task in batch]
        assert isinstance(batch[0], OrderedDict)
        keys = list(batch[0].keys())
        out_dict = OrderedDict()
        for key in keys:
            out_dict[key] = [x[key] for x in batch]
        return out_dict

    return _collate_fn

def no_collate(batch):
    return batch

class ListMetaDataLoader(MetaDataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None):
        collate_fn = batch_list_collate(default_collate)

        super(ListMetaDataLoader, self).__init__(dataset,
            batch_size=batch_size, shuffle=shuffle, sampler=None,
            batch_sampler=None, num_workers=num_workers,
            collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last,
            timeout=timeout, worker_init_fn=worker_init_fn)
