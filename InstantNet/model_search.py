import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
import numpy as np
from thop import profile
from matplotlib import pyplot as plt
from thop import profile
from quantize import QConv2d, QLinear


# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    
    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, layer_id, stride=1, num_bits_list=[32,]):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self.layer_id = layer_id

        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, layer_id, stride, num_bits_list)
            self._ops.append(op)

    def forward(self, x, alpha, num_bits):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            result = result + op(x, num_bits) * w 
            # print(type(op), result.shape)
        return result


    def forward_flops(self, size, alpha):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0

        for w, op in zip(alpha, self._ops):
            flops, size_out = op.forward_flops(size)
            result = result + flops * w

        return result, size_out


class FBNet(nn.Module):
    def __init__(self, config):
        super(FBNet, self).__init__()

        self.num_classes = config.num_classes

        self.num_bits_list = config.num_bits_list

        self.num_layer_list = config.num_layer_list
        self.num_channel_list = config.num_channel_list
        self.stride_list = config.stride_list

        self.stem_channel = config.stem_channel
        self.header_channel = config.header_channel

        self.stem = ConvNorm(3, self.stem_channel, kernel_size=3, stride=1, padding=1, bias=False, num_bits_list=[32,])

        self.cells = nn.ModuleList()

        layer_id = 1

        for stage_id, num_layer in enumerate(self.num_layer_list):
            for i in range(num_layer):
                if i == 0:
                    if stage_id == 0:
                        op = MixedOp(self.stem_channel, self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                    else:
                        op = MixedOp(self.num_channel_list[stage_id-1], self.num_channel_list[stage_id], layer_id, stride=self.stride_list[stage_id], num_bits_list=self.num_bits_list)
                else:
                    op = MixedOp(self.num_channel_list[stage_id], self.num_channel_list[stage_id], layer_id, stride=1, num_bits_list=self.num_bits_list)
                
                layer_id += 1
                self.cells.append(op)


        self.header = ConvNorm(self.num_channel_list[-1], self.header_channel, kernel_size=1, num_bits_list=[32,])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = QLinear(self.header_channel, self.num_classes)

        self._arch_params = self._build_arch_parameters()
        self._reset_arch_parameters()

        self._criterion = nn.CrossEntropyLoss().cuda()

        self.sample_func = config.sample_func


    def forward(self, input, num_bits=32, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)
    
        out = self.stem(input, num_bits=32)

        for i, cell in enumerate(self.cells):
            out = cell(out, alpha[i], num_bits)

        out = self.fc(self.avgpool(self.header(out, num_bits=32)).view(out.size(0), -1), num_bits=32)

        return out
        ###################################
    
    def forward_flops(self, size, temp=1):
        if self.sample_func == 'softmax':
            alpha = F.softmax(getattr(self, "alpha"), dim=-1)
        else:
            alpha = gumbel_softmax(getattr(self, "alpha"), temperature=temp)

        flops_total = []

        flops, size = self.stem.forward_flops(size)
        flops_total.append(flops)

        for i, cell in enumerate(self.cells):
            flops, size = cell.forward_flops(size, alpha[i])
            flops_total.append(flops)

        flops, size = self.header.forward_flops(size)
        flops_total.append(flops)

        return sum(flops_total)


    def _loss_backward(self, input, target, num_bits_list=None, bit_schedule='joint'):
        if num_bits_list is None:
            num_bits_list = self.num_bits_list

        loss_val = [-1 for _ in num_bits_list]

        if bit_schedule == 'joint':
            for num_bits in num_bits_list:
                logit = self(input, num_bits)
                loss = self._criterion(logit, target)
                loss.backward()

                loss_val[num_bits_list.index(num_bits)] = loss.item()

        elif bit_schedule == 'max_loss':
            loss_list = []

            for i, num_bits in enumerate(num_bits_list):
                logit = self(input, num_bits)
                loss = self._criterion(logit, target)

                loss_list.append(loss.item())

                del logit
                del loss

            num_bits_max = num_bits_list[np.array(loss_list).argmax()]

            logit = self(input, num_bits_max)
            loss = self._criterion(logit, target)

            loss.backward()
            loss_val[num_bits_list.index(num_bits_max)] = loss.item()

        return loss_val


    def _build_arch_parameters(self):
        num_ops = len(PRIMITIVES)
        setattr(self, 'alpha', nn.Parameter(Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)))

        return {"alpha": self.alpha}


    def _reset_arch_parameters(self):
        num_ops = len(PRIMITIVES)

        getattr(self, "alpha").data = Variable(1e-3*torch.ones(sum(self.num_layer_list), num_ops).cuda(), requires_grad=True)


    def clip(self):
        for line in getattr(self, "alpha"):
            max_index = line.argmax()
            line.data.clamp_(0, 1)
            if line.sum() == 0.0:
                line.data[max_index] = 1.0
            line.data.div_(line.sum())


if __name__ == '__main__':
    model = FBNet(num_classes=10)
    print(model)