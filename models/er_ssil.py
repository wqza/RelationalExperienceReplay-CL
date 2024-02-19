# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.ResNet18 import resnet18


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Erssil(ContinualModel):
    NAME = 'er_ssil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Erssil, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.seen_classes = [0]
        self.CURRENT_TASK = 0
        self.old_net = None
        self.T = 2

        self.N_CLASSES_PER_TASK = 2
        self.N_TASKS = 5

    def observe(self, inputs, labels, not_aug_inputs):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        batch_size = inputs.shape[0]
        end = self.seen_classes[-1]
        mid = self.seen_classes[-2]

        self.opt.zero_grad()
        if self.CURRENT_TASK > 0:
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            inputs = torch.cat((inputs, buf_inputs))
            # labels_cat = torch.cat((labels, buf_labels))

        outputs = self.net(inputs)
        # loss = self.loss(outputs, labels)
        labels0 = labels % (end - mid)
        loss_CE_curr = self.loss(outputs[:batch_size, mid:end], labels0, reduction='sum')
        if self.CURRENT_TASK > 0:
            # import pdb; pdb.set_trace()
            loss_CE_prev = self.loss(outputs[batch_size:, :mid], buf_labels, reduction='sum')
            loss_CE = (loss_CE_curr + loss_CE_prev) / inputs.size(0)

            # loss_KD
            score = self.old_net(inputs)[:, :mid].data
            loss_KD = torch.zeros(self.CURRENT_TASK).cuda()
            for t in range(self.CURRENT_TASK):
                start_KD = self.seen_classes[t]
                end_KD = self.seen_classes[t + 1]

                soft_target = F.softmax(score[:, start_KD:end_KD] / self.T, dim=1)
                output_log = F.log_softmax(outputs[:, start_KD:end_KD] / self.T, dim=1)
                loss_KD[t] = F.kl_div(output_log, soft_target, reduction='batchmean') * (self.T ** 2)
            loss_KD = loss_KD.sum()
            loss = loss_CE + loss_KD
        else:
            loss = loss_CE_curr / batch_size
        loss.backward()
        self.opt.step()
        # self.buffer.add_data(examples=not_aug_inputs,
        #                      labels=labels)

        return loss.item()

    def end_task(self, dataset):
        # self.old_net = deepcopy(self.net.eval())
        self.old_net = resnet18(self.N_CLASSES_PER_TASK * self.N_TASKS).to(self.device)
        self.old_net.load_state_dict(self.net.state_dict())
        self.fill_buffer(self.buffer, dataset, self.CURRENT_TASK)

    def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
        samples_per_class = mem_buffer.buffer_size // len(self.classes_so_far)
        if t_idx > 0:
            # 1) First, subsample prior classes
            buf_x, buf_y = self.buffer.get_all_data()

            mem_buffer.empty()
            for _y in buf_y.unique():
                idx = (buf_y == _y)
                _y_x, _y_y = buf_x[idx], buf_y[idx]
                mem_buffer.add_data(
                    examples=_y_x[:samples_per_class],
                    labels=_y_y[:samples_per_class]
                )

        # 2) Then, fill with current tasks
        loader = dataset.not_aug_dataloader(self.args.batch_size)

        a_x, a_y = [], []
        for x, y, not_norm_x in loader:
            x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
            a_x.append(not_norm_x.to('cpu'))
            a_y.append(y.to('cpu'))
        a_x, a_y = torch.cat(a_x), torch.cat(a_y)

        for _y in a_y.unique():
            idx = (a_y == _y)
            _x, _y = a_x[idx], a_y[idx]

            randind = torch.randperm(_x.size(0))[:samples_per_class]

            mem_buffer.add_data(
                examples=_x[randind].to(self.device),
                labels=_y[randind].to(self.device)
            )
        assert len(mem_buffer.examples) <= mem_buffer.buffer_size
