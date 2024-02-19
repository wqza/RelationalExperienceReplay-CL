# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.perm_mnist import PermutedMNIST
from datasets.seq_mnist import SequentialMNIST
from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.rot_mnist import RotatedMNIST
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.mnist_360 import MNIST360
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace

from datasets.seq_cifar10_imb import SequentialCIFAR10IMB
from datasets.seq_cifar10_noise import SequentialCIFAR10NOISE
from datasets.seq_cifar10_partial import SequentialCIFAR10PARTIAL0, SequentialCIFAR10PARTIAL1, SequentialCIFAR10PARTIAL2

from datasets.seq_20news import Sequential20News
from datasets.seq_dbpedia import SequentialDBpedia

NAMES = {
    PermutedMNIST.NAME: PermutedMNIST,
    SequentialMNIST.NAME: SequentialMNIST,
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    RotatedMNIST.NAME: RotatedMNIST,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    MNIST360.NAME: MNIST360,
    SequentialCIFAR10IMB.NAME: SequentialCIFAR10IMB,
    SequentialCIFAR10NOISE.NAME: SequentialCIFAR10NOISE,
    SequentialCIFAR10PARTIAL0.NAME: SequentialCIFAR10PARTIAL0,
    SequentialCIFAR10PARTIAL1.NAME: SequentialCIFAR10PARTIAL1,
    SequentialCIFAR10PARTIAL2.NAME: SequentialCIFAR10PARTIAL2,
    Sequential20News.NAME: Sequential20News,
    SequentialDBpedia.NAME: SequentialDBpedia,
}

GCL_NAMES = {
    MNIST360.NAME: MNIST360
}


def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
