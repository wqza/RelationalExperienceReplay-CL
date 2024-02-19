
from utils.buffer import Buffer
from torch.nn import functional as F
from models.utils.continual_model import ContinualModel
from utils.args import *
import torch
import numpy as np
import os
from backbone.MetaResNet18 import resnet18, MetaMergeNet, MetaMLP, MetaClassifier


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


class RDER(ContinualModel):
    NAME = 'RDER'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(RDER, self).__init__(backbone, loss, args, transform)

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

        meta_model = MetaMergeNet(3, 16, 3)
        return meta_model

    def observe(self, inputs, labels, not_aug_inputs):
        # Update the meta model.
        if self.CURRENT_TASK > 0 and (self.i + 1) % self.interval == 0:
            _, loss_val = self.meta_update(train_input=inputs, train_target=labels)
            self.meta_optimizer.zero_grad()
            loss_val.backward()
            self.meta_optimizer.step()

        self.opt.zero_grad()
        outputs = self.net(inputs)
        embed_norm = torch.norm(outputs, dim=1)
        loss = F.cross_entropy(outputs, labels, reduction='none')

        if not self.buffer.is_empty():
            buf_inputs1, _, buf_logits1 = self.buffer.get_data(inputs.shape[0], transform=self.transform)
            buf_inputs2, buf_labels2, _ = self.buffer.get_data(inputs.shape[0], transform=self.transform)

            buf_outputs1 = self.net(buf_inputs1)
            buf_outputs2 = self.net(buf_inputs2)
            embed_norm_buf1 = torch.norm(buf_outputs1, dim=1)
            embed_norm_buf2 = torch.norm(buf_outputs2, dim=1)

            loss_mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
            loss_ce = F.cross_entropy(buf_outputs2, buf_labels2, reduction='none')

            # Generate the paired sample weights by the meta net.
            if self.CURRENT_TASK > 0 and self.epoch > self.warm_start:
                input_meta1 = torch.cat((loss.view(-1, 1),
                                         torch.mean(loss_mse, dim=1).view(-1, 1),
                                         loss_ce.view(-1, 1)), dim=1)
                input_meta2 = torch.cat((embed_norm.unsqueeze(1),
                                         embed_norm_buf1.unsqueeze(1),
                                         embed_norm_buf2.unsqueeze(1)), dim=1)
                with torch.no_grad():
                    weight = self.meta_net(input_meta1.detach(), input_meta2.detach())
            else:
                weight = torch.tensor([[1, self.args.alpha, self.args.beta]]).repeat(inputs.shape[0], 1).to(self.device)

            loss = torch.mean(loss * weight[:, 0]) + \
                   torch.mean(loss_mse * weight[:, 1].unsqueeze(1)) + \
                   torch.mean(loss_ce * weight[:, 2])

        loss = torch.mean(loss)
        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return loss.item()

    def meta_update(self, train_input, train_target):
        if train_input.shape[0] > self.buffer.examples.shape[0]:
            train_input = train_input[:self.buffer.examples.shape[0]]
            train_target = train_target[:self.buffer.examples.shape[0]]
        if self.args.dataset == 'seq-dbpedia':
            temp_net = MetaClassifier(in_dim=768, out_dim=14).to(self.device)
        else:
            temp_net = resnet18(self.net.state_dict()['classifier.bias'].shape[0], num_tasks=self.N_TASKS).to(self.device)
        temp_net.load_state_dict(self.net.state_dict())

        # 1. One step updating the temporary main net.
        outputs = temp_net(train_input)
        embed_norm = torch.norm(outputs, dim=1)
        loss = F.cross_entropy(outputs, train_target.long(), reduction='none')

        buf_inputs1, _, buf_logits1 = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_inputs2, buf_labels2, _ = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_outputs1 = temp_net(buf_inputs1)
        buf_outputs2 = temp_net(buf_inputs2)
        embed_norm_buf1 = torch.norm(buf_outputs1, dim=1)
        embed_norm_buf2 = torch.norm(buf_outputs2, dim=1)
        loss_mse = F.mse_loss(buf_outputs1, buf_logits1, reduction='none')
        loss_ce = F.cross_entropy(buf_outputs2, buf_labels2, reduction='none')

        input_meta1 = torch.cat((loss.view(-1, 1),
                                 torch.mean(loss_mse, dim=1).view(-1, 1),
                                 loss_ce.view(-1, 1)), dim=1)
        input_meta2 = torch.cat((embed_norm.unsqueeze(1),
                                 embed_norm_buf1.unsqueeze(1),
                                 embed_norm_buf2.unsqueeze(1)), dim=1)
        weight = self.meta_net(input_meta1.detach(), input_meta2.detach())

        loss_meta = torch.mean(loss * weight[:, 0]) + \
                    torch.mean(loss_mse * weight[:, 1].unsqueeze(1)) + \
                    torch.mean(loss_ce * weight[:, 2])

        temp_net.zero_grad()
        grads = torch.autograd.grad(loss_meta, temp_net.params(), create_graph=True)
        temp_net.update_params(lr_inner=self.opt.param_groups[0]['lr'], source_params=grads)
        del grads

        # 2. One step updating the meta net.
        buf_inputs, _, buf_logits = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_outputs = temp_net(buf_inputs)
        loss_mse = F.mse_loss(buf_outputs, buf_logits)

        buf_inputs, buf_labels, _ = self.buffer.get_data(train_input.shape[0], transform=self.transform)
        buf_outputs = temp_net(buf_inputs)
        loss_ce = F.cross_entropy(buf_outputs, buf_labels)

        loss_val = loss_mse + loss_ce

        return temp_net, loss_val





