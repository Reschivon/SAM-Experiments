import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class SuperVIN(nn.Module):
    def __init__(self, l_i, l_h, l_q):
        super(SuperVIN, self).__init__()
        self.h = nn.Conv2d(
            in_channels=l_i,
            out_channels=l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        self.conv = nn.Conv2d(l_q, 1, kernel_size=3, stride=1,
                     padding=1, bias=False)

        self.bb = BasicBlock(inplanes=1 + l_q, planes=l_q)

    def forward(self, input_view):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)           # Reward
        q = self.q(r)           # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        def eval_q(r, v):
            input = torch.cat([r, v], 1)
            return self.bb(input)

        # Update q and v values
        k = 50
        for i in range(k - 1):
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        out = self.conv(q)


        return out

class VIN(nn.Module):
    def __init__(self, l_i, l_h, l_q):
        super(VIN, self).__init__()
        self.h = nn.Conv2d(
            in_channels=l_i,
            out_channels=l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

        self.conv = nn.Conv2d(l_q, 1, kernel_size=3, stride=1,
                     padding=1, bias=False)

    def forward(self, input_view):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)           # Reward
        q = self.q(r)           # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        def eval_q(r, v):
            return F.conv2d(
                # Stack reward with most recent value
                torch.cat([r, v], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)

        # Update q and v values
        k = 50
        for i in range(k - 1):
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        out = self.conv(q)


        return out
