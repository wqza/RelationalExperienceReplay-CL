# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from backbone.MetaResNet18 import MetaMLP20news
import torch.nn.functional as F
from utils.conf import data_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader, get_all_trained_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline


def _prepare_twentynews_data(root):
    # https://github.com/morning-dews/PCL/blob/main/helper/twentynews.py
    tnews_train = fetch_20newsgroups(data_home=root, subset='train', remove=('headers', 'footers', 'quotes'))
    tnews_test = fetch_20newsgroups(data_home=root, subset='test', remove=('headers', 'footers', 'quotes'))
    train_texts = tnews_train['data']
    train_y = tnews_train['target']
    test_texts = tnews_test['data']
    test_y = tnews_test['target']

    vectorizer = TfidfVectorizer(max_features=2000)
    train_x = vectorizer.fit_transform(train_texts).todense()
    test_x = vectorizer.transform(test_texts).todense()

    return train_x, train_y, test_x, test_y


class TwentyNews(Dataset):
    def __init__(self, root, vectorizer, train='train', transform=None, target_transform=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        origin_data = fetch_20newsgroups(data_home=root, subset=train, remove=('headers', 'footers', 'quotes'))
        if train == 'train':
            self.data = np.array(vectorizer.fit_transform(origin_data['data']).todense()).astype(float)
        elif train == 'test':
            self.data = np.array(vectorizer.transform(origin_data['data']).todense()).astype(float)
        self.targets = np.array(origin_data['target']).astype(int)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.reshape(-1, 1)
        not_aug_img = img.copy().reshape(1, -1, 1)

        if self.transform is not None:
            img = self.transform(img)
            img = img.to(torch.float32)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train == 'train':
            return img, target, not_aug_img
        else:
            return img, target


class Sequential20News(ContinualDataset):

    NAME = 'seq-20news'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 4
    N_TASKS = 5
    TRANSFORM = transforms.Compose([transforms.ToTensor()])
    vectorizer = TfidfVectorizer(max_features=2000)

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = TwentyNews(data_path() + 'twenty_newsgroups', self.vectorizer,
            train='train', transform=transform)
        test_dataset = TwentyNews(data_path() + 'twenty_newsgroups', self.vectorizer,
            train='test', transform=test_transform)

        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = TwentyNews(data_path() + 'twenty_newsgroups',
            train='train', transform=transform)
        train_loader = get_previous_train_loader(train_dataset, batch_size, self)

        return train_loader

    # @staticmethod
    def get_transform(self):
        # transform = self.TRANSFORM
        # transform = transforms.Compose([self.get_normalization_transform()])
        transform = None
        return transform

    @staticmethod
    def get_backbone():
        return MetaMLP20news(in_dim=2000, out_dim=20)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    # @staticmethod
    # def get_normalization_transform():
    #     transform = transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                                      (0.2470, 0.2435, 0.2615))
    #     return transform
    #
    # @staticmethod
    # def get_denormalization_transform():
    #     transform = DeNormalize((0.4914, 0.4822, 0.4465),
    #                             (0.2470, 0.2435, 0.2615))
    #     return transform
