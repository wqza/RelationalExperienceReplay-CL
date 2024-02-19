# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import sys
conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from argparse import ArgumentParser
from datetime import datetime
from utils.args import add_management_args
from utils.continual_training import train as ctrain
from models import get_model
from utils import create_if_not_exists
from utils.best_args import best_args
from utils.conf import set_random_seed
from utils.conf import base_path
from utils.metrics import *

import torch
from utils.status import progress_bar, create_stash
from utils.tb_logger import *
from utils.loggers import *
from utils.loggers import CsvLogger
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset

import pdb
import time


def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)


def main():
    lecun_fix()
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)

    if args.model == 'mer':
        setattr(args, 'batch_size', 1)

    if args.gpu_id >= 0:
        from backbone.MetaResNet18 import get_gpu_id
        get_gpu_id(gpu_id=args.gpu_id)  # 单gpu训练

    dataset = get_dataset(args)
    backbone = dataset.get_backbone()
    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())

    if isinstance(dataset, ContinualDataset):
        train(model, dataset, args)
    else:
        assert not hasattr(model, 'end_task') or model.NAME == 'joint_gcl'
        ctrain(args)


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    status = model.net.training
    model.net.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            
            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

        accs.append(correct / total * 100 if 'class-il' in model.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    model.net.train(status)
    return accs, accs_mask_classes


def train(model: ContinualModel, dataset: ContinualDataset,
          args: Namespace) -> None:
    """
    The training process, including evaluations and loggers.
    :param model: the module to be trained
    :param dataset: the continual dataset at hand
    :param args: the arguments of the current execution
    """
    save_path = os.path.join(os.getcwd(), 'results', model.NAME + '-' + dataset.NAME + '-' + args.exp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model.net.to(model.device)
    results, results_mask_classes = [], []

    model_stash = create_stash(model, args, dataset)

    if args.csv_log:
        csv_logger = CsvLogger(dataset.SETTING, dataset.NAME, model.NAME)
    if args.tensorboard:
        tb_logger = TensorboardLogger(args, dataset.SETTING, model_stash)
        model_stash['tensorboard_name'] = tb_logger.get_name()

    dataset_copy = get_dataset(args)
    for t in range(dataset.N_TASKS):
        model.net.train()
        _, _ = dataset_copy.get_data_loaders()
    if model.NAME != 'icarl' and model.NAME != 'pnn':
        random_results_class, random_results_task = evaluate(model, dataset_copy)

    model.N_CLASSES_PER_TASK = dataset.N_CLASSES_PER_TASK
    model.N_TASKS = dataset.N_TASKS

    print(file=sys.stderr)
    all_accuracy_cls, all_accuracy_tsk = [], []
    all_forward_cls, all_forward_tsk = [], []
    all_backward_cls, all_backward_tsk = [], []
    all_forgetting_cls, all_forgetting_tsk = [], []
    for t in range(dataset.N_TASKS):
        model.CURRENT_TASK = t
        model.SEEN_CLASS = (t + 1) * model.N_CLASSES_PER_TASK

        model.net.train()
        # continual learning
        train_loader, test_loader = dataset.get_data_loaders()
        if hasattr(model, 'begin_task'):
            model.begin_task(dataset)
        if t:
            accs = evaluate(model, dataset, last=True)
            results[t-1] = results[t-1] + accs[0]
            if dataset.SETTING == 'class-il':
                results_mask_classes[t-1] = results_mask_classes[t-1] + accs[1]
        
        for epoch in range(args.n_epochs):
            model.epoch = epoch

            avg_loss = []
            for i, data in enumerate(train_loader):
                model.i = i

                if hasattr(dataset.train_loader.dataset, 'logits'):
                    inputs, labels, not_aug_inputs, logits = data
                    inputs = inputs.to(model.device)
                    labels = labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    logits = logits.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs, logits)
                else:
                    inputs, labels, not_aug_inputs = data
                    inputs, labels = inputs.to(model.device), labels.to(model.device)
                    not_aug_inputs = not_aug_inputs.to(model.device)
                    loss = model.observe(inputs, labels, not_aug_inputs)
                    time_record['time_iter'][t].append(time_iter2 - time_iter1)

                progress_bar(i, len(train_loader), epoch, t, loss)
                avg_loss.append(loss)

                if args.tensorboard:
                    tb_logger.log_loss(loss, args, epoch, t, i)

                model_stash['batch_idx'] = i + 1

            if hasattr(model, 'end_epoch'):
                model.end_epoch(dataset)

            model_stash['epoch_idx'] = epoch + 1
            model_stash['batch_idx'] = 0

        model_stash['task_idx'] = t + 1
        model_stash['epoch_idx'] = 0

        if hasattr(model, 'end_task'):
            model.end_task(dataset)

        accs = evaluate(model, dataset)
        results.append(accs[0])
        results_mask_classes.append(accs[1])

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        print('class-il:', accs[0], '\ntask-il:', accs[1])

        # print the fwt, bwt, forgetting
        fwt = forward_transfer(results, random_results_class)
        fwt_mask_classes = forward_transfer(results_mask_classes, random_results_task)
        bwt = backward_transfer(results)
        bwt_mask_classes = backward_transfer(results_mask_classes)
        forget = forgetting(results)
        forget_mask_classes = forgetting(results_mask_classes)
        print('Forward: class-il: {}\ttask-il:{}'.format(fwt, fwt_mask_classes))
        print('Backward: class-il: {}\ttask-il:{}'.format(bwt, bwt_mask_classes))
        print('Forgetting: class-il: {}\ttask-il:{}'.format(forget, forget_mask_classes))

        # record the results
        all_accuracy_cls.append(accs[0])
        all_accuracy_tsk.append(accs[1])
        all_forward_cls.append(fwt)
        all_forward_tsk.append(fwt_mask_classes)
        all_backward_cls.append(bwt)
        all_backward_tsk.append(bwt_mask_classes)
        all_forgetting_cls.append(forget)
        all_forgetting_tsk.append(forget_mask_classes)

        model_stash['mean_accs'].append(mean_acc)
        if args.csv_log:
            csv_logger.log(mean_acc)
        if args.tensorboard:
            tb_logger.log_accuracy(np.array(accs), mean_acc, args, t)

        # save model
        if args.save_model:
            if hasattr(model, 'meta_net'):
                torch.save({'task': t + 1,
                            'net.state_dict': model.net.state_dict(),
                            'meta_net.state_dict': model.meta_net.state_dict()},
                           os.path.join(save_path, 'model-task{}.pth'.format(t+1)))
            else:
                torch.save({'task': t + 1,
                            'net.state_dict': model.net.state_dict()},
                           os.path.join(save_path, 'model-task{}.pth'.format(t+1)))

    # record the results
    with open(os.path.join(save_path, 'record.txt'), 'a') as f:
        f.write('\n== 1. Acc:\n==== 1.1. Class-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_cls[t]).strip('[').strip(']') + '\n')
        f.write('\n==== 1.2. Task-IL:\n')
        for t in range(dataset.N_TASKS):
            f.write(str(all_accuracy_tsk[t]).strip('[').strip(']') + '\n')

        f.write('\n== 2. Forward:')
        f.write('\n==== 2.1. Class-IL:\n' + str(all_forward_cls).strip('[').strip(']'))
        f.write('\n==== 2.2. Task-IL:\n' + str(all_forward_tsk).strip('[').strip(']') + '\n')

        f.write('\n== 3. Backward:')
        f.write('\n==== 3.1. Class-IL:\n' + str(all_backward_cls).strip('[').strip(']'))
        f.write('\n==== 3.2. Task-IL:\n' + str(all_backward_tsk).strip('[').strip(']') + '\n')

        f.write('\n== 4. Forgetting:')
        f.write('\n==== 4.1. Class-IL:\n' + str(all_forgetting_cls).strip('[').strip(']'))
        f.write('\n==== 4.2. Task-IL:\n' + str(all_forgetting_tsk).strip('[').strip(']') + '\n')

    if args.csv_log:
        csv_logger.add_bwt(results, results_mask_classes)
        csv_logger.add_forgetting(results, results_mask_classes)
        if model.NAME != 'icarl' and model.NAME != 'pnn':
            csv_logger.add_fwt(results, random_results_class,
                               results_mask_classes, random_results_task)

    if args.tensorboard:
        tb_logger.close()
    if args.csv_log:
        csv_logger.write(vars(args))


if __name__ == '__main__':
    main()
