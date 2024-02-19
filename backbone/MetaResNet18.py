# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List
import math
from torch.autograd import Variable
import torch.nn.init as init


if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'


def get_gpu_id(gpu_id):
    global DEVICE
    DEVICE = DEVICE.replace('0', str(gpu_id))


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.to(DEVICE)
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)

        else:
            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super(MetaBatchNorm2d, self).__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)

        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(MetaBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, MetaLinear) or isinstance(m, MetaConv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(MetaModule):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = MetaBatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = MetaBatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                MetaConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                MetaBatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(MetaModule):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int, num_tasks=1) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = MetaBatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = MetaLinear(nf * 8 * block.expansion, num_classes)
        # self.task_linear = MetaLinear(nf * 8 * block.expansion, num_tasks)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

        self.embedding = None

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor):
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2])  # 512, 1, 1
        self.embedding = out.view(out.size(0), -1)  # 512
        out = self.linear(self.embedding)
        return out
        # return out, feature.detach()  # 必须要detach，否则会保留feature的计算图，导致CUDA显存累积
        # tasks = self.task_linear(feature)
        # # return out, tasks
        # return out, tasks, feature.detach()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def resnet18(nclasses: int, nf: int=64, num_tasks: int=1) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :param num_tasks: number of tasks (by: WQZA)
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf, num_tasks=num_tasks)


class MetaNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(MetaNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        out = self.linear2(x)
        return torch.sigmoid(out)


class MetaMergeNet(MetaModule):
    def __init__(self, input, hidden1, output):
        super(MetaMergeNet, self).__init__()
        self.linear11 = MetaLinear(input, hidden1)
        self.linear12 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden1, output)
        # self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x1, x2):
        x11 = self.linear11(x1)
        x12 = self.linear12(x2)
        x11 = self.relu1(x11)
        x12 = self.relu1(x12)
        out = self.linear2(x11 + x12)
        return torch.sigmoid(out)


class MetaNet_logit(MetaModule):
    def __init__(self, input, hidden1, output):
        super(MetaNet_logit, self).__init__()
        self.linear11 = MetaLinear(input, hidden1)
        self.linear12 = MetaLinear(input, hidden1)
        self.linear13 = MetaLinear(input, hidden1)
        self.relu = nn.ReLU(inplace=True)

        self.linear2 = MetaLinear(hidden1, output)

    def forward(self, x):
        # first layer
        # logits or embeddings
        x11 = self.relu(self.linear11(x[0]))
        x12 = self.relu(self.linear12(x[1]))
        x13 = self.relu(self.linear13(x[2]))

        # second layer
        out = self.linear2(x11 + x12 + x13)
        return torch.sigmoid(out)


class MetaNet_logit_embed(MetaModule):
    def __init__(self, input1, input2, hidden1, hidden2, output):
        super(MetaNet_logit_embed, self).__init__()
        self.linear111 = MetaLinear(input1, hidden1)
        self.linear112 = MetaLinear(input1, hidden1)
        self.linear113 = MetaLinear(input1, hidden1)

        self.linear121 = MetaLinear(input2, hidden1)
        self.linear122 = MetaLinear(input2, hidden1)
        self.linear123 = MetaLinear(input2, hidden1)
        self.relu = nn.ReLU(inplace=True)

        self.linear21 = MetaLinear(hidden1, hidden2)
        self.linear22 = MetaLinear(hidden1, hidden2)

        self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x1, x2):
        # first layer
        # logits
        x11 = self.relu(self.linear111(x1[0]))
        x12 = self.relu(self.linear112(x1[1]))
        x13 = self.relu(self.linear113(x1[2]))
        # embeddings
        x21 = self.relu(self.linear121(x2[0]))
        x22 = self.relu(self.linear122(x2[1]))
        x23 = self.relu(self.linear123(x2[2]))

        # second layer
        out11 = self.relu(self.linear21(x11 + x12 + x13))
        out12 = self.relu(self.linear22(x21 + x22 + x23))

        # third layer
        out = self.linear3(out11 + out12)
        return torch.sigmoid(out)


class MetaNet_logit_loss(MetaModule):
    def __init__(self, input1, hidden1, hidden2, output):
        super(MetaNet_logit_loss, self).__init__()
        self.linear111 = MetaLinear(input1, hidden1)
        self.linear112 = MetaLinear(input1, hidden1)
        self.linear113 = MetaLinear(input1, hidden1)

        self.linear12 = MetaLinear(3, hidden1)
        self.relu = nn.ReLU(inplace=True)

        self.linear21 = MetaLinear(hidden1, hidden2)
        self.linear22 = MetaLinear(hidden1, hidden2)

        self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x1, x2):
        # first layer
        # logits
        x11 = self.relu(self.linear111(x1[0]))
        x12 = self.relu(self.linear112(x1[1]))
        x13 = self.relu(self.linear113(x1[2]))
        # loss
        x2 = self.relu(self.linear12(x2))

        # second layer
        out11 = self.relu(self.linear21(x11 + x12 + x13))
        out12 = self.relu(self.linear22(x2))

        # third layer
        out = self.linear3(out11 + out12)
        return torch.sigmoid(out)


class MetaNet_logit_embed_loss(MetaModule):
    def __init__(self, input1, input2, hidden1, hidden2, output):
        super(MetaNet_logit_embed_loss, self).__init__()
        self.linear111 = MetaLinear(input1, hidden1)
        self.linear112 = MetaLinear(input1, hidden1)
        self.linear113 = MetaLinear(input1, hidden1)

        self.linear121 = MetaLinear(input2, hidden1)
        self.linear122 = MetaLinear(input2, hidden1)
        self.linear123 = MetaLinear(input2, hidden1)

        self.linear13 = MetaLinear(3, hidden1)
        self.relu = nn.ReLU(inplace=True)

        self.linear21 = MetaLinear(hidden1, hidden2)
        self.linear22 = MetaLinear(hidden1, hidden2)
        self.linear23 = MetaLinear(hidden1, hidden2)

        self.linear3 = MetaLinear(hidden2, output)

    def forward(self, x1, x2, x3):
        # first layer
        # logits
        x11 = self.relu(self.linear111(x1[0]))
        x12 = self.relu(self.linear112(x1[1]))
        x13 = self.relu(self.linear113(x1[2]))
        # embddings
        x21 = self.relu(self.linear121(x2[0]))
        x22 = self.relu(self.linear122(x2[1]))
        x23 = self.relu(self.linear123(x2[2]))
        # loss
        x3 = self.relu(self.linear13(x3))

        # second layer
        out11 = self.relu(self.linear21(x11 + x12 + x13))
        out12 = self.relu(self.linear22(x21 + x22 + x23))
        out13 = self.relu(self.linear23(x3))

        # third layer
        out = self.linear3(out11 + out12 + out13)
        return torch.sigmoid(out)


class LSTMCell(MetaModule):

    def __init__(self, num_inputs, hidden_size):
        super(LSTMCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = MetaLinear(num_inputs, 4 * hidden_size)
        self.fc_h2h = MetaLinear(hidden_size, 4 * hidden_size)

    def init_weights(self):
        initrange = 0.1
        self.fc_h2h.weight.data.uniform_(-initrange, initrange)
        self.fc_i2h.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)
        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)

        in_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1] - 1)
        out_gate = torch.sigmoid(gates[2])
        in_transform = torch.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * torch.tanh(cx)
        return hx, cx


class MLRSNetCell(MetaModule):

    def __init__(self, num_inputs, hidden_size):
        super(MLRSNetCell, self).__init__()

        self.hidden_size = hidden_size
        self.fc_i2h = nn.Sequential(
            MetaLinear(num_inputs, hidden_size),
            nn.ReLU(),
            MetaLinear(hidden_size, 4 * hidden_size)
        )
        self.fc_h2h = nn.Sequential(
            MetaLinear(hidden_size, hidden_size),
            nn.ReLU(),
            MetaLinear(hidden_size, 4 * hidden_size)
        )
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for module in self.fc_h2h:
            if type(module) == MetaLinear:
                module.weight.data.uniform_(-initrange, initrange)
        for module in self.fc_i2h:
            if type(module) == MetaLinear:
                module.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs, state):
        hx, cx = state
        i2h = self.fc_i2h(inputs)
        h2h = self.fc_h2h(hx)

        x = i2h + h2h
        gates = x.split(self.hidden_size, 1)

        in_gate = torch.sigmoid(gates[0])
        forget_gate = torch.sigmoid(gates[1] - 1)
        out_gate = torch.sigmoid(gates[2])
        in_transform = torch.tanh(gates[3])

        cx = forget_gate * cx + in_gate * in_transform
        hx = out_gate * torch.tanh(cx)
        return hx, cx


class MLRSNet(MetaModule):

    def __init__(self, num_layers, hidden_size, in_dim=1, out_dim=1):
        super(MLRSNet, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layer1 = MLRSNetCell(in_dim, hidden_size)
        self.layer2 = nn.Sequential(*[MLRSNetCell(hidden_size, hidden_size) for _ in range(num_layers-1)])
        self.layer3 = MetaLinear(hidden_size, out_dim)

    def reset_lstm(self, keep_states=False, device='cpu'):
        if keep_states:
            for i in range(len(self.layer2)+1):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
                self.hx[i], self.cx[i] = self.hx[i].to(device), self.cx[i].to(device)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.layer2) + 1):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.hx[i], self.cx[i] = self.hx[i].to(device), self.cx[i].to(device)

    def forward(self, x):
        if x.size(0) != self.hx[0].size(0):
            self.hx[0] = self.hx[0].expand(x.size(0), self.hx[0].size(1))
            self.cx[0] = self.cx[0].expand(x.size(0), self.cx[0].size(1))
        self.hx[0], self.cx[0] = self.layer1(x, (self.hx[0], self.cx[0]))
        x = self.hx[0]

        for i in range(1, self.num_layers):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.layer2[i-1](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        out = self.layer3(x)
        # out = torch.sigmoid(out)
        return out


class MetaClassifier(MetaModule):
    def __init__(self, in_dim, out_dim):
        super(MetaClassifier, self).__init__()
        self.fc = MetaLinear(in_dim, out_dim)

        for m in self.modules():
            if isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = x.squeeze()  # for DBpedia
        return self.fc(x)


class MetaMLP(MetaModule):
    def __init__(self, in_dim, out_dim, hid_dim=256):
        super(MetaMLP, self).__init__()
        self.linear1 = MetaLinear(in_dim, hid_dim)
        self.linear2 = MetaLinear(hid_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.squeeze()  # for DBpedia
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class MetaMLP20news(MetaModule):
    def __init__(self, in_dim, out_dim, hid_dim=256):
        super(MetaMLP20news, self).__init__()
        self.linear1 = MetaLinear(in_dim, hid_dim)
        self.linear2 = MetaLinear(hid_dim, hid_dim)
        self.linear3 = MetaLinear(hid_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()

        for m in self.modules():
            if isinstance(m, MetaLinear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, MetaBatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x = x.to(torch.float32)  # for 20news
        x = x.squeeze()  # for DBpedia, 20news
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)

        x = self.linear3(x)
        return x