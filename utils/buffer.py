# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from torchvision import transforms


def icarl_replay(self, dataset, val_set_split=0):
    """
    Merge the replay buffer with the current task data.
    Optionally split the replay buffer into a validation set.
    :param self: the model instance
    :param dataset: the dataset
    :param val_set_split: the fraction of the replay buffer to be used as validation set
    """

    if self.task > 0:
        buff_val_mask = torch.rand(len(self.buffer)) < val_set_split
        val_train_mask = torch.zeros(len(dataset.train_loader.dataset.data)).bool()
        val_train_mask[torch.randperm(len(dataset.train_loader.dataset.data))[:buff_val_mask.sum()]] = True

        if val_set_split > 0:
            self.val_loader = deepcopy(dataset.train_loader)

        data_concatenate = torch.cat if isinstance(dataset.train_loader.dataset.data, torch.Tensor) else np.concatenate
        need_aug = hasattr(dataset.train_loader.dataset, 'not_aug_transform')
        if not need_aug:
            def refold_transform(x): return x.cpu()
        else:
            data_shape = len(dataset.train_loader.dataset.data[0].shape)
            if data_shape == 3:
                def refold_transform(x): return (x.cpu() * 255).permute([0, 2, 3, 1]).numpy().astype(np.uint8)
            elif data_shape == 2:
                def refold_transform(x): return (x.cpu() * 255).squeeze(1).type(torch.uint8)

        # REDUCE AND MERGE TRAINING SET
        dataset.train_loader.dataset.targets = np.concatenate([
            dataset.train_loader.dataset.targets[~val_train_mask],
            self.buffer.labels.cpu().numpy()[:len(self.buffer)][~buff_val_mask]
        ])
        dataset.train_loader.dataset.data = data_concatenate([
            dataset.train_loader.dataset.data[~val_train_mask],
            refold_transform((self.buffer.examples)[:len(self.buffer)][~buff_val_mask])
        ])

        if val_set_split > 0:
            # REDUCE AND MERGE VALIDATION SET
            self.val_loader.dataset.targets = np.concatenate([
                self.val_loader.dataset.targets[val_train_mask],
                self.buffer.labels.cpu().numpy()[:len(self.buffer)][buff_val_mask]
            ])
            self.val_loader.dataset.data = data_concatenate([
                self.val_loader.dataset.data[val_train_mask],
                refold_transform((self.buffer.examples)[:len(self.buffer)][buff_val_mask])
            ])


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class Buffer:
    """
    The memory buffer of rehearsal method.
    """
    def __init__(self, buffer_size, device, n_tasks=None, mode='reservoir'):
        assert mode in ['ring', 'reservoir']
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_examples = 0
        self.functional_index = eval(mode)
        if mode == 'ring':
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ['examples', 'labels', 'logits', 'task_labels']

        # self.examples_dict = {}
        # self.labels_dict = {}
        # self.logits_dict = {}
        # self.task_labels_dict = {}
        self.examples_list = []
        self.labels_list = []
        self.logits_list = []
        self.task_labels_list = []
        self.num_seen_examples_per_class = []

    def init_tensors(self, examples: torch.Tensor, labels: torch.Tensor,
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:
        """
        Initializes just the required tensors.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        """
        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith('els') else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))

    def add_data(self, examples, labels=None, logits=None, task_labels=None):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                self.examples[index] = examples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)

    def add_data_by_sample_weights(self, sample_weights, examples, labels=None, logits=None, task_labels=None,
                                   sample_method='equal'):
        """
        Adds the data to the memory buffer according to the reservoir strategy.
        :param examples: tensor containing the images
        :param labels: tensor containing the labels
        :param logits: tensor containing the outputs of the network
        :param task_labels: tensor containing the task labels
        :return:
        """
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        index_list = []
        for i in range(examples.shape[0]):
            index = reservoir(self.num_seen_examples, self.buffer_size)
            self.num_seen_examples += 1
            if index >= 0:
                index_list.append(index)
        replacing_num = len(index_list)  # the index in index_list would be replaced by incoming examples.
        if sample_method == 'max':
            # the index of the largest replacing_num samples in the sample_weights
            replacing_index = np.argsort(-np.array(sample_weights))[:replacing_num]
        elif sample_method == 'min':
            # the index of the smallest replacing_num samples in the sample_weights
            replacing_index = np.argsort(np.array(sample_weights))[:replacing_num]
        elif sample_method == 'equal':
            # the index of the smallest replacing_num samples in the sample_weights
            equal_idx = np.around(np.linspace(0, examples.shape[0], replacing_num, endpoint=False)).astype(int)
            replacing_index = np.argsort(np.array(sample_weights))[equal_idx]
        elif sample_method == 'random':
            # the index of the smallest replacing_num samples in the sample_weights
            replacing_index = np.random.choice(len(sample_weights), replacing_num, replace=False)
        else:
            raise ValueError('Unexpected sample method.')

        self.examples[index_list] = examples[replacing_index].to(self.device)
        if labels is not None:
            self.labels[index_list] = labels[replacing_index].to(self.device)
        if logits is not None:
            self.logits[index_list] = logits[replacing_index].to(self.device)
        if task_labels is not None:
            self.task_labels[index_list] = task_labels[replacing_index].to(self.device)

    def equal_add_data(self, examples, labels=None, logits=None, task_labels=None, num_classes=1):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)

        buffer_partition_size = [self.buffer_size // num_classes for i in range(num_classes)]
        for i in range(self.buffer_size % num_classes):
            buffer_partition_size[i] += 1

        past_cls_num = len(self.examples_list)
        if past_cls_num < num_classes:
            # if new classes come in:
            # 1. reduce the buffer of past classes to new partition size
            for cls in range(past_cls_num):
                self.examples_list[cls] = self.examples_list[cls][:buffer_partition_size[cls]]
                if labels is not None:
                    self.labels_list[cls] = self.labels_list[cls][:buffer_partition_size[cls]]
                if logits is not None:
                    self.logits_list[cls] = self.logits_list[cls][:buffer_partition_size[cls]]
                if task_labels is not None:
                    self.task_labels_list[cls] = self.task_labels_list[cls][:buffer_partition_size[cls]]

            # 2. initialize the buffer
            for cls in range(past_cls_num, num_classes):
                self.examples_list.append(torch.zeros(buffer_partition_size[cls], examples.shape[1], examples.shape[2], examples.shape[3]))
                if labels is not None:
                    self.labels_list.append(torch.zeros(buffer_partition_size[cls]))
                if logits is not None:
                    self.logits_list.append(torch.zeros(buffer_partition_size[cls], logits.shape[1]))
                if task_labels is not None:
                    self.task_labels_list.append(torch.zeros(buffer_partition_size[cls]))
                self.num_seen_examples_per_class.append(0)

        for i in range(examples.shape[0]):
            cls = labels[i]
            # import pdb; pdb.set_trace()
            self.num_seen_examples_per_class[cls] += 1
            index = reservoir(self.num_seen_examples_per_class[cls], buffer_partition_size[cls])
            if index >= 0:
                self.examples_list[cls][index] = examples[i]
                if labels is not None:
                    self.labels_list[cls][index] = labels[i]
                if logits is not None:
                    self.logits_list[cls][index] = logits[i]
                if task_labels is not None:
                    self.task_labels_list[cls][index] = task_labels[i]

        self.examples = torch.cat(self.examples_list).to(self.device)
        if labels is not None:
            self.labels = torch.cat(self.labels_list).to(self.device).long()
        if logits is not None:
            self.logits = torch.cat(self.logits_list).to(self.device)
        if task_labels is not None:
            self.task_labels = torch.cat(self.task_labels_list).to(self.device).long()
        self.num_seen_examples = sum(self.num_seen_examples_per_class)

    def offline_add_data(self, examples, labels=None, logits=None, task_labels=None, drop_idx=None):
        if not hasattr(self, 'examples'):
            self.init_tensors(examples, labels, logits, task_labels)
        # import pdb; pdb.set_trace()
        if drop_idx is None:
            drop_idx = np.random.choice(np.arange(self.buffer_size), examples.shape[0], replace=False)
        for i in range(len(drop_idx)):
            try:
                self.examples[drop_idx[i]] = examples[i].to(self.device)
            except IndexError:
                import pdb; pdb.set_trace()
            if labels is not None:
                self.labels[drop_idx[i]] = labels[i].to(self.device)
            if logits is not None:
                self.logits[drop_idx[i]] = logits[i].to(self.device)
            if task_labels is not None:
                self.task_labels[drop_idx[i]] = task_labels[i].to(self.device)

    def get_data(self, size: int, transform: transforms=None, cls=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        # import pdb; pdb.set_trace()
        if cls is None:
            if size > min(self.num_seen_examples, self.examples.shape[0]):
                size = min(self.num_seen_examples, self.examples.shape[0])

            choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=size, replace=False)
            if transform is None: transform = lambda x: x
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice],)
        else:
            idx_cls = torch.where(self.labels == cls)
            idx_cls = np.array(idx_cls[0].cpu())
            # if size > len(idx_cls):
            #     size = len(idx_cls)

            try:
                choice = np.random.choice(idx_cls, size, replace=False)
            except ValueError:
                import pdb; pdb.set_trace()
            if transform is None: transform = lambda x: x
            ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
            for attr_str in self.attributes[1:]:
                if hasattr(self, attr_str):
                    attr = getattr(self, attr_str)
                    ret_tuple += (attr[choice],)

        return ret_tuple

    def get_equal_data(self, size: int, transform: transforms=None) -> Tuple:
        """
        Random samples a batch of size items.
        :param size: the number of requested items
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        # import pdb; pdb.set_trace()
        if size > min(self.num_seen_examples, self.examples.shape[0]):
            size = min(self.num_seen_examples, self.examples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]), size=size, replace=False)
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)

        return ret_tuple

    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False

    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu()) for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple

    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0
