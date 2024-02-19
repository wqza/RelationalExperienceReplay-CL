# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from backbone.MetaResNet18 import MetaMLP, MetaClassifier
import torch.nn.functional as F
from utils.conf import data_path
from PIL import Image
import os
from datasets.utils.validation import get_train_val
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders
from datasets.utils.continual_dataset import get_previous_train_loader, get_all_trained_loaders
from typing import Tuple
from datasets.transforms.denormalization import DeNormalize


class DBpedia(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 classes=None) -> None:
        self.not_aug_transform = transforms.Compose([transforms.ToTensor()])
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes

        self.data = []
        self.targets = []

        if self.classes is None:
            self.classes = range(14)

        if train:
            for cls in self.classes:
                xx = torch.load(os.path.join(root, 'train_data{}.pt'.format(cls + 1)))  # np.array
                self.data.append(xx)
                self.targets.append(np.ones(xx.shape[0]) * cls)
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets).astype(int)
        else:
            xx = torch.load(os.path.join(root, 'test_data.pt'))
            yy = torch.load(os.path.join(root, 'test_label.pt'))
            for cls in self.classes:
                self.data.append(xx[yy == cls + 1])
                self.targets.append(yy[yy == cls + 1] - 1)
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = img.reshape(-1, 1)
        not_aug_img = img.copy().reshape(1, -1, 1)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.train:
            return img, target, not_aug_img
        else:
            return img, target


class SequentialDBpedia(ContinualDataset):

    NAME = 'seq-dbpedia'
    SETTING = 'class-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 7
    TRANSFORM = transforms.Compose([transforms.ToTensor()])
    task = 0

    def get_data_loaders(self):
        transform = self.TRANSFORM
        test_transform = transforms.Compose([transforms.ToTensor()])

        classes = np.arange(self.i, self.i + self.N_CLASSES_PER_TASK)
        train_dataset = DBpedia(data_path() + 'DBpedia-processed',
            train=True, transform=transform, classes=classes)
        test_dataset = DBpedia(data_path() + 'DBpedia-processed',
            train=False, transform=test_transform, classes=classes)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)

        self.i += self.N_CLASSES_PER_TASK
        self.train_loader = train_loader
        self.test_loaders.append(test_loader)
        return train_loader, test_loader

    def not_aug_dataloader(self, batch_size):
        transform = transforms.Compose([transforms.ToTensor()])

        classes = np.arange(SequentialDBpedia.task * SequentialDBpedia.N_CLASSES_PER_TASK,
                            (SequentialDBpedia.task + 1) * SequentialDBpedia.N_CLASSES_PER_TASK)
        train_dataset = DBpedia(data_path() + 'DBpedia-processed',
            train=True, transform=transform, classes=classes)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        return train_loader

    # @staticmethod
    def get_transform(self):
        # transform = self.TRANSFORM
        # transform = transforms.Compose([self.get_normalization_transform()])
        transform = None
        return transform

    @staticmethod
    def get_backbone():
        # return MetaMLP(in_dim=768, out_dim=14)
        return MetaClassifier(in_dim=768, out_dim=14)

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
