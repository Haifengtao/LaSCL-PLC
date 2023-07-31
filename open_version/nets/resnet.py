#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   resnet.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, 2021

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/8/8 10:10   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
# aa= models()

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, droprate=0, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop = nn.Dropout3d(droprate)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_cls_classes,
                 num_seg_classes=2,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, num_cls_classes)
        self.drop1d = nn.Dropout3d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(
                512 * block.expansion,
                32,
                2,
                stride=2
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_seg_classes,
                kernel_size=1,
                stride=(1, 1, 1),
                bias=False)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        concat = []
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x = self.relu(x)
        concat.append(x)
        x = self.maxpool(x)  # [b, 64, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        concat.append(x)
        # x = self.drop3d(x)
        x = self.layer2(x)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.drop3d(x)
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        x = self.drop3d(x)
        # for cls
        # pdb.set_trace()
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.drop1d(x)
        x = self.finel_fc(x)

        # for seg
        # x = self.conv_seg(x)

        return x


class Resnet_seg(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes=2,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_seg, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, num_cls_classes)
        self.drop1d = nn.Dropout3d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.conv_seg1 = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))

        self.conv_seg2 = nn.Sequential(nn.ConvTranspose3d(256 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(1),
                                       nn.ReLU(inplace=True))

        self.conv_seg4 = nn.Sequential(nn.Conv3d(2, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 2, kernel_size=1, stride=(1, 1, 1),
                                                 bias=False),
)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        inpt = x
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        x = self.drop3d(x2)

        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.drop1d(x)
        x = self.finel_fc(x)

        # for seg
        # pdb.set_trace()o
        seg = self.conv_seg1(x2)
        seg = self.conv_seg2(torch.cat((seg, x1), dim=1))
        seg = self.conv_seg3(torch.cat((seg, x0), dim=1))
        seg = self.conv_seg4(torch.cat((seg, inpt), dim=1))

        return x, seg

    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        return x


class Resnet_seg2cls(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_seg2cls, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, num_cls_classes)
        self.drop1d = nn.Dropout3d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.conv_seg1 = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))

        self.conv_seg2 = nn.Sequential(nn.ConvTranspose3d(256 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, num_seg_classes, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # inpt = x)
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        x = self.drop3d(x2)

        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.drop1d(x)
        x = self.finel_fc(x)

        # for seg
        # pdb.set_trace()o
        seg = self.conv_seg1(x2)
        seg = self.conv_seg2(torch.cat((seg, x1), dim=1))
        seg = self.conv_seg3(torch.cat((seg, x0), dim=1))

        return seg

    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        return x


class Resnet_seg_moco(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes=2,
                 proj_dim=128,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_seg_moco, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, proj_dim*2)

        self.proj_fc = nn.Linear(proj_dim*2, proj_dim)
        self.drop1d = nn.Dropout1d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.conv_seg1 = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))

        self.conv_seg2 = nn.Sequential(nn.ConvTranspose3d(256 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))


        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 2, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        inpt = x
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        x = self.drop3d(x2)

        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        x = self.proj_fc(x)
        # for seg
        # pdb.set_trace()o
        seg = self.conv_seg1(x2)
        seg = self.conv_seg2(torch.cat((seg, x1), dim=1))
        seg = self.conv_seg3(torch.cat((seg, x0), dim=1))
        return x, seg

    @ torch.no_grad()
    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        return x


class Resnet_vae_moco(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes=2,
                 proj_dim=128,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_vae_moco, self).__init__()
        self.conv1 = nn.Conv3d(1,64,kernel_size=7,stride=(2, 2, 2),padding=(3, 3, 3),bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d((2,2,2))

        self.mu_fc = nn.Linear(512 * block.expansion * 8, proj_dim*2)
        self.var_fc = nn.Linear(512 * block.expansion * 8, proj_dim*2)

        self.proj_fc = nn.Linear(proj_dim*2, proj_dim)
        self.drop1d = nn.Dropout3d(droprate)
        self.drop3d = nn.Dropout3d(droprate)

        self.decoder_input = nn.Linear(proj_dim*2, 512 * block.expansion * 8)
        self.conv_decoder1 = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 512, 2, stride=2),
                                           nn.BatchNorm3d(512),
                                           nn.ReLU(inplace=True),
                                           nn.Conv3d(512, 512, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                     bias=False),
                                           nn.BatchNorm3d(512),
                                           nn.ReLU(inplace=True),)

        self.conv_decoder2 = nn.Sequential(nn.ConvTranspose3d(512, 256, 2, stride=2),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(256, 256, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(256),
                                       nn.ReLU(inplace=True))


        self.conv_decoder3 = nn.Sequential(nn.ConvTranspose3d(256, 64, 2, stride=2),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(64, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 1, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def reparameterize(self, mu, logvar):
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        # encoder
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]


        # for KL
        x = self.avgpool(x2)  # [b, 2048, 2, 2, 2]
        x = x.flatten(1)
        mu = self.mu_fc(x)
        log_var = self.var_fc(x)
        z = self.reparameterize(mu, log_var)

        # for CL
        q = self.proj_fc(z)
        q_mu = self.proj_fc(mu)

        # for recon
        # pdb.set_trace()
        result = self.decoder_input(z)
        result = result.view(-1, 2048, 2, 2, 2)
        # import pdb
        # pdb.set_trace()
        result = F.interpolate(result, size=(12, 14, 8), mode='trilinear', align_corners=None)
        recon = self.conv_decoder1(result)
        recon = self.conv_decoder2(recon)
        recon = self.conv_decoder3(recon)
        return [q, q_mu], recon, mu, log_var

    @ torch.no_grad()
    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        mu = self.mu_fc(x)
        # x = self.proj_fc(mu)
        return mu


class Resnet_raw_moco(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes=2,
                 proj_dim=128,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_raw_moco, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, 256)
        self.proj_fc = nn.Linear(256, proj_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        inpt = x
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # x = self.drop3d(x2)

        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        x = self.proj_fc(x)
        return x, 0

    @ torch.no_grad()
    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.finel_fc(x)
        return x



class ResNet10_Segment(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 droprate=0,
                 shortcut_type='B',
                 no_cuda=False):
        super(ResNet10_Segment, self).__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.finel_fc = nn.Linear(512 * block.expansion, num_cls_classes)
        self.drop1d = nn.Dropout3d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.conv_seg1 = nn.Sequential(nn.ConvTranspose3d(512 * block.expansion, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seg2 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 64, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(64),
                                       nn.ReLU(inplace=True))

        self.conv_seg3 = nn.Sequential(nn.ConvTranspose3d(64 * 2, 32, 2, stride=2),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 2, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False))

        self.conv_seg4 = nn.Sequential(nn.Conv3d(2, 32, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1),
                                                 bias=False),
                                       nn.BatchNorm3d(32),
                                       nn.ReLU(inplace=True),
                                       nn.Conv3d(32, 2, kernel_size=1, stride=(1, 1, 1),
                                                 bias=False),
)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False),
                    nn.BatchNorm3d(planes * block.expansion)
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        inpt = x
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 64, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 128, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 256, x/8, y/8, z/8]
        x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 512, x/8, y/8, z/8]
        x = self.drop3d(x2)

        # for cls
        x = self.avgpool(x)  # [b, 512, 1, 1, 1]
        x = x.flatten(1)
        x = self.drop1d(x)
        x = self.finel_fc(x)

        # for seg
        # pdb.set_trace()o
        seg = self.conv_seg1(x2)
        seg = self.conv_seg2(torch.cat((seg, x1), dim=1))
        seg = self.conv_seg3(torch.cat((seg, x0), dim=1))
        # import pdb
        # pdb.set_trace()
        # seg = self.conv_seg4(torch.cat((seg, inpt), dim=1))

        return x, seg

    def get_hidden_vec(self, x):
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]
        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        # x = self.drop3d(x)
        x = self.layer2(x1)  # [b, 512, x/8, y/8, z/8]
        # x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]
        # for cls
        x = self.avgpool(x)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        return x


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet10_seg(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet10_Segment(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_seg(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_seg(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_seg2cls(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_seg2cls(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet50_seg_moco(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_seg_moco(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet50_raw_moco(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_raw_moco(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet50_vae_moco(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_vae_moco(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


if __name__ == '__main__':
    import pdb, os

    pth_dir = "../external/pretrain/resnet_10.pth"
    new_layer_names = ['upsample1', 'cmp_layer3', 'upsample2', 'cmp_layer2', 'upsample3', 'cmp_layer1', 'upsample4',
                       'cmp_conv1',
                       'conv_seg']
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    x = torch.rand((1, 1, 96, 112, 64)).cuda()
    model = resnet50_vae_moco(num_cls_classes=2,
                             num_seg_classes=2,
                             ).cuda()
    # net_dict = model.state_dict()
    #
    # print('loading pretrained model {}'.format(pth_dir))
    # pretrain = torch.load(pth_dir)
    # # for k, v in pretrain['state_dict'].items():
    # #     # pdb.set_trace()
    # #     if k[7:] in net_dict.keys():
    # #         print(k)
    #
    # pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
    # net_dict.update(pretrain_dict)
    # model.load_state_dict(net_dict)
    # new_parameters = []
    # for pname, p in model.named_parameters():
    #     for layer_name in new_layer_names:
    #         if pname.find(layer_name) >= 0:
    #             new_parameters.append(p)
    #             break
    #
    # new_parameters_id = list(map(id, new_parameters))
    # base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
    # parameters = {'base_parameters': base_parameters,
    #               'new_parameters': new_parameters}
    y = model(x)
    pdb.set_trace()
