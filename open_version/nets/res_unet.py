#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   res_unet.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/18 18:46   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from functools import partial

class Dilate_block(nn.Module):
    def __init__(self, channel):
        super(Dilate_block, self).__init__()
        self.dilate1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1),
            nn.LeakyReLU()
        )
        self.dilate2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2),
            nn.LeakyReLU()
        )
        self.dilate3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4),
            nn.LeakyReLU()
        )
        self.dilate4 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8),
            nn.LeakyReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        # dilate5_out = self.dilate5(dilate4_out)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel, drop_rate=0):
        super(DoubleConv, self).__init__()
        if drop_rate > 0:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.Dropout2d(drop_rate),
                nn.LeakyReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.Dropout2d(drop_rate),
                nn.LeakyReLU(),
            )

        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.LeakyReLU(),
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(outchannel),
                nn.LeakyReLU(),
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Cls_layer(nn.Module):
    def __init__(self):
        super(Cls_layer, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d()
        self.avg_pool = nn.AdaptiveAvgPool2d()
    def forward(self, x):
        pass

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

class Res_Unet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3, drop_rate=0):
        super(Res_Unet34, self).__init__()

        filters = [64, 128, 256, 512, 512]
        resnet = models.resnet34(pretrained=False)
        self.inputs = nn.Sequential(
            nn.Conv2d(num_channels, filters[0], kernel_size=7, stride=2, padding=3, bias=False),
                      resnet.bn1,
                      resnet.relu)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dilate_block(512)

        self.doubleConv5 = DoubleConv(filters[4], filters[3], drop_rate=drop_rate)
        self.doubleConv6 = DoubleConv(filters[3], filters[2], drop_rate=drop_rate)
        self.doubleConv7 = DoubleConv(filters[2], filters[1], drop_rate=drop_rate)
        self.doubleConv8 = DoubleConv(filters[1], filters[0], drop_rate=drop_rate)
        self.up1 = nn.ConvTranspose2d(filters[4], filters[4] // 2, 2, stride=(2, 2))
        self.up2 = nn.ConvTranspose2d(filters[3], filters[3] // 2, 2, stride=(2, 2))
        self.up3 = nn.ConvTranspose2d(filters[2], filters[2] // 2, 2, stride=(2, 2))
        self.up4 = nn.ConvTranspose2d(filters[1], filters[1] // 2, 2, stride=(2, 2))

        self.out = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1),
            # nn.Softmax()
        )

    def forward(self, x):
        # Encoder
        x = self.inputs(x)             # 64, w, h
        e1 = self.encoder1(x)          # 64, w//2, h//2
        e2 = self.encoder2(e1)         # 128, w//2, h//2
        e3 = self.encoder3(e2)         # 256, w//2, h//2
        e4 = self.encoder4(e3)         # 512, w//2, h//2

        # Center
        e4 = self.dblock(e4)

        # Decoder
        x = self.up1(e4)
        x = self.doubleConv5(torch.cat([e4, x], 1))
        x = self.up2(x)
        x = self.doubleConv6(torch.cat([e3, x], 1))
        x = self.up3(x)
        x = self.doubleConv7(torch.cat([e2, x], 1))
        x = self.up4(x)
        x = self.doubleConv8(torch.cat([e1, x], 1))
        x = self.out(x)
        x = self.out(x)

        return x


class Resnet_MED_3D(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_cls_classes,
                 num_seg_classes=2,
                 proj_dim=128,
                 droprate=0.5,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(Resnet_MED_3D, self).__init__()
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
        self.final_fc = nn.Linear(512 * block.expansion, proj_dim*2)
        self.drop1d = nn.Dropout1d(droprate)
        self.drop3d = nn.Dropout3d(droprate)
        self.cls_head = nn.Sequential(nn.Linear(258, 128), nn.ReLU(), nn.Linear(128, 2))


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

    def forward(self, x, age, sex):
        inpt = x
        # import pdb
        # pdb.set_trace()
        x = self.conv1(x)  # [b, 64, x/2, y/2, z/2]
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)  # [b, 64, x/4, y/4, z/4]

        x1 = self.layer1(x)  # [b, 256, x/4, y/4, z/4]
        x = self.drop3d(x1)
        x = self.layer2(x)  # [b, 512, x/8, y/8, z/8]
        x = self.drop3d(x)
        x = self.layer3(x)  # [b, 1024, x/8, y/8, z/8]
        x = self.drop3d(x)
        x2 = self.layer4(x)  # [b, 2048, x/8, y/8, z/8]

        # for cls
        x = self.avgpool(x2)  # [b, 2048, 1, 1, 1]
        x = x.flatten(1)
        x = self.drop1d(x)
        x = self.final_fc(x)

        # for seg
        seg = self.conv_seg1(x2)
        seg = self.conv_seg2(torch.cat((seg, x1), dim=1))
        seg = self.conv_seg3(torch.cat((seg, x0), dim=1))


        q = nn.functional.normalize(x, dim=1)
        multi_q = torch.cat([q, torch.unsqueeze(age, dim=-1), torch.unsqueeze(sex, dim=-1)], dim=-1)
        res = self.cls_head(multi_q)
        return res, seg

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



def seg_MED_3D(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Resnet_MED_3D(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

if __name__ == '__main__':
    model = seg_MED_3D(num_cls_classes=2)
    x = torch.randn((1,1, 256, 256, 256))
    age = torch.tensor([0.56])
    sex = torch.tensor([0])
    y = model(x, age, sex)
    import pdb
    print(y.size())
