
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
import os
import pickle
from backbone.MetaResNet18 import resnet18, MetaMergeNet


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Penalty weight.')
    return parser


class RERACEspeed(ContinualModel):
    NAME = 'RERACEspeed'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(RERACEspeed, self).__init__(backbone, loss, args, transform)

        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.meta_net = self.create_meta_model()
        self.meta_optimizer = torch.optim.Adam(self.meta_net.params(), lr=1e-3, weight_decay=1e-4)

        self.epoch = 0  # current epoch
        self.i = 0  # iteration in every epoch
        self.N_TASKS = 0
        self.N_CLASSES_PER_TASK = 0
        self.CURRENT_TASK = 0

        self.warm_start = np.floor(args.n_epochs / 2)
        self.interval = np.floor(args.n_epochs / 10)

    def create_meta_model(self):
        if self.args.gpu_id >= 0:
            from backbone.MetaResNet18 import get_gpu_id
            get_gpu_id(gpu_id=self.args.gpu_id)  # single gpu

        meta_model = MetaMergeNet(2, 16, 2)
        return meta_model

    def observe(self, inputs, labels, not_aug_inputs):
        # Update the meta model.
        if self.CURRENT_TASK > 0 and (self.i + 1) % self.interval == 0:
            _, loss_val = self.meta_update(train_input=inputs, train_target=labels)
            self.meta_optimizer.zero_grad()
            loss_val.backward()
            self.meta_optimizer.step()

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()

        if self.CURRENT_TASK > 0:
            present = labels.unique().long()
            outputs = self.net(inputs)
            mask = torch.zeros_like(outputs)
            mask[:, present] = 1
            # unmask unseen classes
            mask[:, self.classes_so_far.max():] = 1

            outputs = outputs.masked_fill(mask == 0, outputs.min())
            buf_inputs, buf_labels = self.buffer.get_data(outputs.shape[0], transform=self.transform)
            buf_outputs = self.net(buf_inputs)

            loss_curr = F.cross_entropy(outputs, labels.long(), reduction='none')
            loss_buf = F.cross_entropy(buf_outputs, buf_labels.long(), reduction='none')
            embed_norm_curr = torch.norm(outputs, dim=1)
            embed_norm_buf = torch.norm(buf_outputs, dim=1)

            # Generate the paired sample weights by the meta net.
            if self.CURRENT_TASK > 0 and self.epoch > self.warm_start:
                input_meta1 = torch.cat((loss_curr.view(-1, 1),
                                         loss_buf.view(-1, 1)), dim=1)
                input_meta2 = torch.cat((embed_norm_curr.unsqueeze(1),
                                         embed_norm_buf.unsqueeze(1)), dim=1)
                with torch.no_grad():
                    weight = self.meta_net(input_meta1.detach(), input_meta2.detach())
            else:
                weight = torch.tensor([[1, 1]]).to(self.device)

            loss = torch.mean(loss_curr * weight[:, 0]) + torch.mean(loss_buf * weight[:, 1])

        else:
            outputs = self.net(inputs)
            loss = F.cross_entropy(outputs, labels.long())

        loss = torch.mean(loss)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()

    def meta_update(self, train_input, train_target):
        if train_input.shape[0] > self.buffer.examples.shape[0]:
            train_input = train_input[:self.buffer.examples.shape[0]]
            train_target = train_target[:self.buffer.examples.shape[0]]
        temp_net = resnet18(self.net.state_dict()['classifier.bias'].shape[0], num_tasks=self.N_TASKS).to(self.device)
        temp_net.load_state_dict(self.net.state_dict())

        # 1. One step updating the temporary main net.
        present = train_target.unique().long()
        outputs = temp_net(train_input)
        mask = torch.zeros_like(outputs)
        mask[:, present] = 1
        # unmask unseen classes
        mask[:, self.classes_so_far.max():] = 1
        outputs = outputs.masked_fill(mask == 0, outputs.min())

        buf_inputs, buf_labels = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_outputs = temp_net(buf_inputs)

        loss_curr = F.cross_entropy(outputs, train_target.long(), reduction='none')
        loss_buf = F.cross_entropy(buf_outputs, buf_labels.long(), reduction='none')

        embed_norm_curr = torch.norm(outputs, dim=1)
        embed_norm_buf = torch.norm(buf_outputs, dim=1)

        input_meta1 = torch.cat((loss_curr.view(-1, 1),
                                 loss_buf.view(-1, 1)), dim=1)
        input_meta2 = torch.cat((embed_norm_curr.unsqueeze(1),
                                 embed_norm_buf.unsqueeze(1)), dim=1)
        with torch.no_grad():
            weight = self.meta_net(input_meta1.detach(), input_meta2.detach())

        loss_meta = torch.mean(loss_curr * weight[:, 0]) + torch.mean(loss_buf * weight[:, 1])

        temp_net.zero_grad()
        grads = torch.autograd.grad(loss_meta, temp_net.linear.params(), create_graph=True)
        temp_net.linear.update_params(lr_inner=self.opt.param_groups[0]['lr'], source_params=grads)
        del grads

        # 2. One step updating the meta net.
        buf_inputs, buf_labels = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_outputs = temp_net(buf_inputs)
        loss_val = F.cross_entropy(buf_outputs, buf_labels)

        return temp_net, loss_val

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





