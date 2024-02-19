# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight.')
    return parser


class Derpp(ContinualModel):
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(Derpp, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.CURRENT_TASK = 0
        self.N_CLASSES_PER_TASK = 0
        self.EPOCH = 0

    def observe(self, inputs, labels, not_aug_inputs):

        self.opt.zero_grad()
        outputs = self.net(inputs)
        loss = self.loss(outputs, labels.long())

        if not self.buffer.is_empty():
            # buf_inputs, _, buf_logits = self.buffer.get_data(
            buf_inputs, _, buf_logits = self.buffer.get_equal_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)
        # import pdb; pdb.set_trace()

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs, labels=labels, logits=outputs.data)
        # self.buffer.equal_add_data(examples=not_aug_inputs,
        #                            labels=labels,
        #                            logits=outputs.data,
        #                            num_classes=(self.CURRENT_TASK + 1) * self.N_CLASSES_PER_TASK)
        # self.buffer.add_data_by_sample_weights(sample_weights=np.arange(inputs.shape[0]) / inputs.shape[0],
        #                                        examples=not_aug_inputs,
        #                                        labels=labels,
        #                                        logits=outputs.data,
        #                                        sample_method='equal')

        return loss.item()

    # def end_task(self, dataset) -> None:
    #     with torch.no_grad():
    #         self.fill_buffer(self.buffer, dataset, self.CURRENT_TASK)
    #     # self.CURRENT_TASK += 1
    #
    # def fill_buffer(self, mem_buffer: Buffer, dataset, t_idx: int) -> None:
    #     """
    #     Adds examples from the current task to the memory buffer
    #     by means of the herding strategy.
    #     :param mem_buffer: the memory buffer
    #     :param dataset: the dataset from which take the examples
    #     :param t_idx: the task index
    #     """
    #
    #     mode = self.net.training
    #     self.net.eval()
    #     # samples_per_class = mem_buffer.buffer_size // self.SEEN_CLASS
    #     samples_per_class = [self.args.buffer_size // self.SEEN_CLASS for i in range(self.SEEN_CLASS)]
    #     for i in range(self.args.buffer_size % self.SEEN_CLASS):
    #         samples_per_class[i] += 1
    #
    #     if t_idx > 0:
    #         # 1) First, subsample prior classes
    #         buf_x, buf_y, buf_l = self.buffer.get_all_data()
    #
    #         mem_buffer.empty()
    #         for i, _y in enumerate(buf_y.unique()):
    #             idx = (buf_y == _y)
    #             _y_x, _y_y, _y_l = buf_x[idx], buf_y[idx], buf_l[idx]
    #             mem_buffer.add_data(
    #                 examples=_y_x[:samples_per_class[i]],
    #                 labels=_y_y[:samples_per_class[i]],
    #                 logits=_y_l[:samples_per_class[i]]
    #             )
    #
    #     # 2) Then, fill with current tasks
    #     loader = dataset.not_aug_dataloader(self.args.batch_size)
    #
    #     # 2.1 Extract all features
    #     a_x, a_y, a_f, a_l = [], [], [], []
    #     for x, y, not_norm_x in loader:
    #         x, y, not_norm_x = (a.to(self.device) for a in [x, y, not_norm_x])
    #         a_x.append(not_norm_x.to('cpu'))
    #         a_y.append(y.to('cpu'))
    #
    #         # features
    #         feats = self.net.features(x)
    #         a_f.append(feats.cpu())
    #         a_l.append(torch.sigmoid(self.net.classifier(feats)).cpu())
    #
    #         # # meta weight net
    #         # with torch.no_grad():
    #         #     out = self.net(x)
    #         #     loss = F.cross_entropy(out, y, reduction='none')
    #         #     embed_norm = torch.norm(out, dim=1)
    #         #     input_meta1 = torch.cat((loss.view(-1, 1),
    #         #                              # torch.mean(loss_mse, dim=1).view(loss_mse.shape[0], 1),
    #         #                              torch.mean(loss).repeat(loss.shape[0]).view(-1, 1)), dim=1)
    #         #     input_meta2 = torch.cat((embed_norm.view(-1, 1),
    #         #                              # embed_norm_buf1.unsqueeze(1),
    #         #                              torch.mean(embed_norm).repeat(embed_norm.shape[0]).view(-1, 1)), dim=1)
    #         #     weight = self.meta_net(input_meta1, input_meta2)
    #         #     weight = weight[:, 0] / weight[:, 1]
    #         #     a_f.append(weight.detach().cpu())
    #         #     a_l.append(torch.sigmoid(out).cpu())
    #     a_x, a_y, a_f, a_l = torch.cat(a_x), torch.cat(a_y), torch.cat(a_f), torch.cat(a_l)
    #
    #     # 2.2 Compute class means
    #     for cls in a_y.unique():
    #         idx = (a_y == cls)
    #         _x, _y, _l = a_x[idx], a_y[idx], a_l[idx]
    #         # feats = a_f[idx]
    #         # mean_feat = feats.mean(0, keepdim=True)
    #         #
    #         # running_sum = torch.zeros_like(mean_feat)
    #         # i = 0
    #         # while i < samples_per_class and i < feats.shape[0]:
    #         #     cost = (mean_feat - (feats + running_sum) / (i + 1)).norm(2, 1)
    #         #     idx_min = cost.argmin().item()
    #         #
    #         #     mem_buffer.add_data(
    #         #         examples=_x[idx_min:idx_min + 1].to(self.device),
    #         #         labels=_y[idx_min:idx_min + 1].to(self.device),
    #         #         logits=_l[idx_min:idx_min + 1].to(self.device)
    #         #     )
    #         #
    #         #     running_sum += feats[idx_min:idx_min + 1]
    #         #     feats[idx_min] = feats[idx_min] + 1e6
    #         #     i += 1
    #
    #         # random sample
    #         index = np.random.choice(np.arange(_x.shape[0]), samples_per_class[cls], replace=False)
    #         mem_buffer.add_data(
    #             examples=_x[index].to(self.device),
    #             labels=_y[index].to(self.device),
    #             logits=_l[index].to(self.device)
    #         )
    #
    #         # # sample by meta weight
    #         # # weight = a_f[idx]
    #         # weight = np.ones(len(_x))
    #         # sample_method = 'equal'
    #         # if sample_method == 'max':
    #         #     # the index of the largest replacing_num samples in the sample_weights
    #         #     index = np.argsort(-np.array(weight))[:samples_per_class[cls]]
    #         # elif sample_method == 'min':
    #         #     # the index of the smallest replacing_num samples in the sample_weights
    #         #     index = np.argsort(np.array(weight))[:samples_per_class[cls]]
    #         # elif sample_method == 'equal':
    #         #     # the index of the smallest replacing_num samples in the sample_weights
    #         #     equal_idx = np.around(np.linspace(0, weight.shape[0], samples_per_class[cls], endpoint=False)).astype(int)
    #         #     index = np.argsort(np.array(weight))[equal_idx]
    #         # else:
    #         #     raise ValueError('Unexpected sample method.')
    #         # mem_buffer.add_data(
    #         #     examples=_x[index].to(self.device),
    #         #     labels=_y[index].to(self.device),
    #         #     logits=_l[index].to(self.device)
    #         # )
    #
    #     assert len(mem_buffer.examples) <= mem_buffer.buffer_size
    #
    #     self.net.train(mode)

    # def end_epoch(self, dataset) -> None:
    #     mode = self.net.training
    #     self.net.eval()
    #
    #     # replace all newly added samples by sample weights
    #     if self.CURRENT_TASK > 0 and self.EPOCH > 0:
    #         new_cls = np.arange(self.N_CLASSES_PER_TASK * self.CURRENT_TASK,
    #                             self.N_CLASSES_PER_TASK * (self.CURRENT_TASK + 1))
    #
    #         loader = dataset.not_aug_dataloader(self.args.batch_size)
    #         x_list, y_list, logits_list = [], [], []
    #         for x, y, not_norm_x in loader:
    #             x_list.append(not_norm_x.cpu())
    #             y_list.append(y.cpu())
    #             x, y, not_norm_x = x.to(self.device), y.to(self.device), not_norm_x.to(self.device)
    #             with torch.no_grad():
    #                 out = self.net(x)
    #                 logits_list.append(out.cpu())
    #         x_list, y_list, logits_list = torch.cat(x_list), torch.cat(y_list), torch.cat(logits_list)
    #         x_list = [x_list[y_list == cls] for cls in new_cls]
    #         # import pdb; pdb.set_trace()
    #         weight_list = [np.ones(len(x_list[cls])) for cls in range(len(x_list))]
    #         logits_list = [logits_list[y_list == cls] for cls in new_cls]
    #
    #         sample_method = 'random'
    #         for i, cls in enumerate(new_cls):
    #             new_buffer_num = sum(self.buffer.labels == cls).item()
    #             # import pdb; pdb.set_trace()
    #             if sample_method == 'max':
    #                 # the index of the largest replacing_num samples in the sample_weights
    #                 index = np.argsort(-np.array(weight_list[i]))[:new_buffer_num]
    #             elif sample_method == 'min':
    #                 index = np.argsort(np.array(weight_list[i]))[:new_buffer_num]
    #             elif sample_method == 'equal':
    #                 equal_idx = np.around(np.linspace(0, weight_list[i].shape[0], new_buffer_num, endpoint=False)).astype(int)
    #                 index = np.argsort(np.array(weight_list[i]))[equal_idx]
    #             elif sample_method == 'random':
    #                 index = np.random.choice(len(weight_list[i]), new_buffer_num, replace=False)
    #             else:
    #                 raise ValueError('Unexpected sample method.')
    #             self.buffer.examples[self.buffer.labels == cls] = x_list[i][index].to(self.device)
    #             self.buffer.logits[self.buffer.labels == cls] = logits_list[i][index].to(self.device)
    #
    #     self.net.train(mode)



