# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import sys
from typing import Dict, Any
from utils.metrics import *

from utils import create_if_not_exists
from utils.conf import base_path
import numpy as np

useless_args = ['dataset', 'tensorboard', 'validation', 'model',
                'csv_log', 'notes', 'load_best_args']


def print_mean_accuracy(mean_acc: np.ndarray, task_number: int,
                        setting: str) -> None:
    """
    Prints the mean accuracy on stderr.
    :param mean_acc: mean accuracy value
    :param task_number: task index
    :param setting: the setting of the benchmark
    """
    if setting == 'domain-il':
        mean_acc, _ = mean_acc
        print('\nAccuracy for {} task(s): {} %'.format(
            task_number, round(mean_acc, 2)), file=sys.stderr)
    else:
        mean_acc_class_il, mean_acc_task_il = mean_acc
        print('\nAccuracy for {} task(s): \t [Class-IL]: {} %'
              ' \t [Task-IL]: {} %\n'.format(task_number, round(
            mean_acc_class_il, 2), round(mean_acc_task_il, 2)), file=sys.stderr)
        # mean_acc_class_il, mean_acc_task_il, mean_acc_pred_task = mean_acc
        # print('\nAccuracy for {} task(s): \t [Class-IL]: {} \t [Task-IL]: {} \t [pred-task]: {} %\n'.
        #       format(task_number, round(mean_acc_class_il, 2), round(mean_acc_task_il, 2), round(mean_acc_pred_task, 2)), file=sys.stderr)


class CsvLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, experiment_id=1) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.accs_cluster = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = model_str
        self.fwt = None
        self.fwt_mask_classes = None
        self.bwt = None
        self.bwt_mask_classes = None
        self.forgetting = None
        self.forgetting_mask_classes = None
        self.experiment_id = experiment_id

    def add_fwt(self, results, accs, results_mask_classes, accs_mask_classes):
        self.fwt = forward_transfer(results, accs)
        if self.setting == 'class-il':
            self.fwt_mask_classes = forward_transfer(results_mask_classes, accs_mask_classes)

    def add_bwt(self, results, results_mask_classes):
        self.bwt = backward_transfer(results)
        self.bwt_mask_classes = backward_transfer(results_mask_classes)

    def add_forgetting(self, results, results_mask_classes):
        self.forgetting = forgetting(results)
        self.forgetting_mask_classes = forgetting(results_mask_classes)

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            mean_acc_class_il, mean_acc_task_il = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)
            # mean_acc_class_il, mean_acc_task_il, mean_acc_cluster = mean_acc
            # self.accs.append(mean_acc_class_il)
            # self.accs_mask_classes.append(mean_acc_task_il)
            # self.accs_cluster.append(mean_acc_cluster)

    def write(self, args: Dict[str, Any]) -> None:
        """
        writes out the logged value along with its arguments.
        :param args: the namespace of the current experiment
        """
        for cc in useless_args:
            if cc in args:
                del args[cc]

        columns = list(args.keys())

        new_cols = []
        for i, acc in enumerate(self.accs):
            args['task' + str(i + 1)] = acc
            new_cols.append('task' + str(i + 1))

        args['forward_transfer'] = self.fwt
        new_cols.append('forward_transfer')

        args['backward_transfer'] = self.bwt
        new_cols.append('backward_transfer')

        args['forgetting'] = self.forgetting
        new_cols.append('forgetting')

        columns = new_cols + columns

        # create_if_not_exists(os.path.join(base_path(), "results", self.setting))
        # create_if_not_exists(os.path.join(base_path(), "results", self.setting, self.dataset))
        # create_if_not_exists(os.path.join(base_path(), "results", self.setting, self.dataset, self.model))
        create_if_not_exists(os.path.join(base_path(), "standard_results", self.setting))
        create_if_not_exists(os.path.join(base_path(), "standard_results", self.setting, self.dataset))
        create_if_not_exists(os.path.join(base_path(), "standard_results", self.setting, self.dataset, self.model))

        write_headers = False
        # path = os.path.join(base_path(), "results", self.setting, self.dataset, self.model, str(self.experiment_id) + "mean_accs.csv")
        path = os.path.join(base_path(), "standard_results", self.setting, self.dataset, self.model, str(self.experiment_id) + "mean_accs.csv")
        if not os.path.exists(path):
            write_headers = True
        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            if write_headers:
                writer.writeheader()
            writer.writerow(args)

        if self.setting == 'class-il':
            # create_if_not_exists(os.path.join(base_path(), "results", "task-il", self.dataset))
            # create_if_not_exists(os.path.join(base_path(), "results", "task-il", self.dataset, self.model))
            create_if_not_exists(os.path.join(base_path(), "standard_results", "task-il", self.dataset))
            create_if_not_exists(os.path.join(base_path(), "standard_results", "task-il", self.dataset, self.model))

            for i, acc in enumerate(self.accs_mask_classes):
                args['task' + str(i + 1)] = acc

            args['forward_transfer'] = self.fwt_mask_classes
            args['backward_transfer'] = self.bwt_mask_classes
            args['forgetting'] = self.forgetting_mask_classes

            write_headers = False
            # path = os.path.join(base_path(), "results", "task-il", self.dataset, self.model, str(self.experiment_id) + "mean_accs.csv")
            path = os.path.join(base_path(), "standard_results", "task-il", self.dataset, self.model, str(self.experiment_id) + "mean_accs.csv")
            if not os.path.exists(path):
                write_headers = True
            with open(path, 'a') as tmp:
                writer = csv.DictWriter(tmp, fieldnames=columns)
                if write_headers:
                    writer.writeheader()
                writer.writerow(args)


class TrainingLogger:
    def __init__(self, setting_str: str, dataset_str: str,
                 model_str: str, buffer_size: int, experiment_id=0) -> None:
        self.accs = []
        if setting_str == 'class-il':
            self.accs_mask_classes = []
            self.accs_cluster = []
        self.setting = setting_str
        self.dataset = dataset_str
        self.model = str(experiment_id) + '_' + model_str + '_' + str(buffer_size)
        self.write_header = True

    def log(self, mean_acc: np.ndarray) -> None:
        """
        Logs a mean accuracy value.
        :param mean_acc: mean accuracy value
        """
        if self.setting == 'general-continual':
            self.accs.append(mean_acc)
        elif self.setting == 'domain-il':
            mean_acc, _ = mean_acc
            self.accs.append(mean_acc)
        else:
            # mean_acc_class_il, mean_acc_task_il = mean_acc
            # self.accs.append(mean_acc_class_il)
            # self.accs_mask_classes.append(mean_acc_task_il)
            mean_acc_class_il, mean_acc_task_il, mean_acc_cluster = mean_acc
            self.accs.append(mean_acc_class_il)
            self.accs_mask_classes.append(mean_acc_task_il)
            self.accs_cluster.append(mean_acc_cluster)

    def write(self, task=1, epoch=0, loss=None, loss_new=None, loss_mse=None, loss_ce=None, loss_task=None, weight=None, acc=None):
        training_logger = {'task': task, 'epoch': epoch}
        if loss is not None:
            training_logger['total_loss'] = round(loss, 6)
        if loss_new is not None:
            training_logger['loss_new'] = round(loss_new, 6)
        if loss_mse is not None:
            training_logger['loss_mse'] = round(loss_mse, 6)
        if loss_ce is not None:
            training_logger['loss_ce'] = round(loss_ce, 6)
        if loss_task is not None:
            training_logger['loss_task'] = round(loss, 6)
        if weight is not None:
            training_logger['weight'] = np.round(weight, 6)
        if acc is not None:
            # import pdb; pdb.set_trace()
            training_logger['Class_IL'] = acc[0]
            training_logger['Task_IL'] = acc[1]
            if len(acc) > 2:
                training_logger['Cluster_IL'] = acc[2]

        # create_if_not_exists(os.path.join(base_path(), "results"))
        create_if_not_exists(os.path.join(base_path(), "standard_results", self.dataset))

        # path = os.path.join(base_path(), "results/", self.model + "_training_logger.csv")
        path = os.path.join(base_path(), "standard_results", self.dataset, self.model + "_training_logger.csv")

        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=training_logger.keys())
            if self.write_header:
                writer.writeheader()
            writer.writerow(training_logger)
        self.write_header = False

    def add_results(self, task, acc, forward, backward):
        training_logger = {'task': task, 'Evaluation': 'True'}
        mean_acc = np.mean(acc, axis=1)
        training_logger['cls_acc'] = acc[0]
        training_logger['cls_mean_acc'] = mean_acc[0]
        training_logger['cls_forward'] = forward[0]
        training_logger['cls_backward'] = backward[0]
        training_logger['tsk_acc'] = acc[1]
        training_logger['tsk_mean_acc'] = mean_acc[1]
        training_logger['tsk_forward'] = forward[1]
        training_logger['tsk_backward'] = backward[1]

        create_if_not_exists(os.path.join(base_path(), "standard_results", self.dataset))
        path = os.path.join(base_path(), "standard_results/", self.dataset, self.model + "_training_logger.csv")

        with open(path, 'a') as tmp:
            writer = csv.DictWriter(tmp, fieldnames=training_logger.keys())
            if self.write_header:
                writer.writeheader()
            writer.writerow(training_logger)
        self.write_header = False
