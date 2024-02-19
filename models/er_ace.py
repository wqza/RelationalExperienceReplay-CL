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


class ErAce(ContinualModel):
    NAME = 'er_ace'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(ErAce, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)
        self.CURRENT_TASK = 0
        # self.seen_cur = 0

    def observe(self, inputs, labels, not_aug_inputs):
        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()
        # self.seen_cur += inputs.size(0)

        if self.CURRENT_TASK > 0:
            present = labels.unique().long()
            outputs = self.net(inputs)
            mask = torch.zeros_like(outputs)
            mask[:, present] = 1
            # unmask unseen classes
            mask[:, self.classes_so_far.max():] = 1

            outputs = outputs.masked_fill(mask == 0, -1e9)
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buf_outputs = self.net(buf_inputs)

            loss = self.loss(outputs, labels.long())/2 + self.loss(buf_outputs, buf_labels.long())/2

        else:
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())

        loss.backward()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels)

        return loss.item()

    # def end_task(self, dataset):
    #     self.seen_cur = 0
