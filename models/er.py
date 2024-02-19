# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle
import numpy as np
import torch
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    return parser


class Er(ContinualModel):
    NAME = 'er'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Er, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Record some observation
        self.START_TRAINING = False
        self.count_new = 0
        self.count_old = 0
        self.record_dict = {}
        self.TOTAL_CLS_NUM = 10

    def observe(self, inputs, labels, not_aug_inputs):

        real_batch_size = inputs.shape[0]
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels)

        self.opt.zero_grad()

        if not self.buffer.is_empty():
            buf_inputs, buf_labels = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            # inputs = torch.cat((inputs, buf_inputs))
            # labels = torch.cat((labels, buf_labels))
            buf_outputs = self.net(buf_inputs)
            buf_loss = self.loss(buf_outputs, buf_labels)
            # loss = loss + 0.5 * buf_loss
            loss = loss + buf_loss
            # # loss /= 2
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size])

        return loss.item()

    # def observe(self, inputs, labels, not_aug_inputs):
    #
    #     real_batch_size = inputs.shape[0]
    #     outputs = self.net(inputs)
    #
    #     self.opt.zero_grad()
    #     loss1 = self.loss(outputs, labels.long())
    #     loss1.backward()
    #     self.opt.step()
    #
    #     # 1. record gradient of new tasks
    #     grad = self.record_dict['gradient-new']
    #     cls_norm = self.record_dict['cls-norm']
    #     embed_norm = self.record_dict['embed-norm']
    #     logits_norm = self.record_dict['logits-norm']
    #     for name, params in self.net.named_parameters():
    #         grad[name] = (grad[name] * self.count_new + torch.norm(params.grad).item()) / (self.count_new + 1)
    #         if name == 'linear.weight':
    #             cls_norm['cls-weight'] = (cls_norm['cls-weight'] * self.count_new + np.array(torch.norm(params, dim=1).detach().cpu())) / (self.count_new + 1)
    #         elif name == 'linear.bias':
    #             cls_norm['cls-bias'] = (cls_norm['cls-bias'] * self.count_new + np.array(params.detach().cpu())) / (self.count_new + 1)
    #     for cls in range(self.TOTAL_CLS_NUM):
    #         cls_idx = torch.where(labels == cls)
    #         cls_idx = cls_idx[0]
    #         embeds = torch.norm(self.net.embedding.detach().cpu(), dim=1)
    #         logits = torch.norm(outputs.detach().cpu(), dim=1)
    #         if len(cls_idx) > 0:
    #             embed_norm[cls] = (embed_norm[cls] * self.count_new + np.array(torch.mean(embeds[cls_idx]))) / (self.count_new + 1)
    #             logits_norm[cls] = (logits_norm[cls] * self.count_new + np.array(torch.mean(logits[cls_idx]))) / (self.count_new + 1)
    #     self.record_dict['gradient-new'] = grad
    #     self.record_dict['cls-norm'] = cls_norm
    #     self.record_dict['embed-norm'] = embed_norm
    #     self.record_dict['logits-norm'] = logits_norm
    #     self.count_new += 1
    #
    #     if not self.buffer.is_empty():
    #         buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
    #         # inputs = torch.cat((inputs, buf_inputs))
    #         # labels = torch.cat((labels, buf_labels))
    #         buf_outputs = self.net(buf_inputs)
    #
    #         self.opt.zero_grad()
    #         loss2 = 0.5 * self.loss(buf_outputs, buf_labels.long())
    #         loss2.backward()
    #         self.opt.step()
    #
    #         loss1 += loss2
    #
    #         # 2. record gradient of buffer samples
    #         grad = self.record_dict['gradient-old']
    #         cls_norm = self.record_dict['cls-norm']
    #         embed_norm = self.record_dict['embed-norm']
    #         logits_norm = self.record_dict['logits-norm']
    #         for name, params in self.net.named_parameters():
    #             grad[name] = (grad[name] * self.count_old + torch.norm(params.grad).item()) / (self.count_old + 1)
    #             if name == 'linear.weight':
    #                 cls_norm['cls-weight'] = (cls_norm['cls-weight'] * self.count_old + np.array(torch.norm(params, dim=1).detach().cpu())) / (self.count_old + 1)
    #             elif name == 'linear.bias':
    #                 cls_norm['cls-bias'] = (cls_norm['cls-bias'] * self.count_old + np.array(params.detach().cpu())) / (self.count_old + 1)
    #         for cls in range(self.TOTAL_CLS_NUM):
    #             cls_idx = torch.where(buf_labels == cls)
    #             cls_idx = cls_idx[0]
    #             embeds = torch.norm(self.net.embedding.detach().cpu(), dim=1)
    #             logits = torch.norm(buf_outputs.detach().cpu(), dim=1)
    #             if len(cls_idx) > 0:
    #                 # 这里没有把新类数据和 buffer 中新类的数据的 embedding 或 logits 分开
    #                 embed_norm[cls] = (embed_norm[cls] * self.count_old + np.array(torch.mean(embeds[cls_idx]))) / (self.count_old + 1)
    #                 logits_norm[cls] = (logits_norm[cls] * self.count_old + np.array(torch.mean(logits[cls_idx]))) / (self.count_old + 1)
    #         self.record_dict['gradient-old'] = grad
    #         self.record_dict['cls-norm'] = cls_norm
    #         self.record_dict['embed-norm'] = embed_norm
    #         self.record_dict['logits-norm'] = logits_norm
    #         self.count_old += 1
    #
    #     self.buffer.add_data(examples=not_aug_inputs,
    #                          labels=labels[:real_batch_size])
    #
    #     return loss1.item()
    #
    # def start_record(self, record_grad=True, record_cls_norm=True, record_embed_norm=True, record_logits_norm=True):
    #     self.count_new = 0
    #     self.count_old = 0
    #     if record_grad:
    #         self.record_dict['gradient-new'] = {}
    #         self.record_dict['gradient-old'] = {}
    #         for name, params in self.net.named_parameters():
    #             self.record_dict['gradient-new'][name] = 0
    #             self.record_dict['gradient-old'][name] = 0
    #             if name == 'linear.bias':
    #                 self.TOTAL_CLS_NUM = len(params)
    #     if record_cls_norm:
    #         self.record_dict['cls-norm'] = {
    #             'cls-weight': np.zeros(self.TOTAL_CLS_NUM),
    #             'cls-bias': np.zeros(self.TOTAL_CLS_NUM)
    #         }
    #     if record_embed_norm:
    #         self.record_dict['embed-norm'] = np.zeros(self.TOTAL_CLS_NUM)
    #     if record_logits_norm:
    #         self.record_dict['logits-norm'] = np.zeros(self.TOTAL_CLS_NUM)
    #
    # def save_record(self, task, epoch):
    #     save_path = os.path.join(self.save_path, 'task{}'.format(task))
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #
    #     if epoch % 5 == 0:
    #         with open(os.path.join(save_path, 'epoch-{}-record.pkl'.format(epoch)), 'wb') as pk:
    #             pickle.dump(self.record_dict, pk)

