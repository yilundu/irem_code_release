from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os

if __name__ == '__main__':
    file_names = []
    with open('maml_split.txt', 'r') as f:
        for row in f:
            row = ['omniglot'] + row.strip().split('/')[-2:]
            file_names.append(row)
    
    data_folder = '../../../dataset/omniglot'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    splits = [file_names[:1100], file_names[1100:1200], file_names[1200:]]
    phases = ['train', 'val', 'test']
    for phase, split in zip(phases, splits):
        with open(os.path.join(data_folder, '%s_labels.json' % phase), 'w') as f:
            json.dump(split, f)
