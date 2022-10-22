from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import torch
import torch.utils.data as data

import multiprocessing
import six
from torch.utils.data import TensorDataset, DataLoader, Dataset, SequentialSampler


def load_all_data(text_file):
    total_data = json.load(open(text_file, 'r'))
    train_data = {}
    val_data = {}
    test_data = {}

    for id, item in total_data.items():
        if item["split"] == 'train':
            train_data[id] = item
        if item["split"] == 'valid':
            val_data[id] = item
        if item["split"] == 'test':
            test_data[id] = item
    print('there are {} data in train, {} data in val, {} data in test'.format(len(list(train_data.keys())), len(list(val_data.keys())), len(list(test_data.keys()))))
    return  train_data, val_data, test_data




class load_ce_data(Dataset):
    def __init__(self, data, img_feature_path):


        self.img_feature_path = img_feature_path

        self.ids = []
        self.texts = []
        self.labels = []
        self.captions = []
        for id, item in data.items():
            id = id
            text = item["text"]
            label = item["label"]
            caption = item['caption']
            self.ids.append(id)
            self.texts.append(text)
            self.captions.append(caption)
            self.labels.append(label)

    def __len__(self):
        return (len(self.ids))

    def __getitem__(self, i):
        id = self.ids[i]
        text = self.texts[i]
        label = self.labels[i]
        caption = self.captions[i]
        img_feature = np.load(os.path.join(self.img_feature_path, id+'.npz'))
        sample = (img_feature, text, label, caption)

        return sample
