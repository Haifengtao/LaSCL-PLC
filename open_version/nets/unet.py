#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   unet.py    
@Contact :   760320171@qq.com
@License :   (C)Copyright, ISTBI

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/7/5 14:43   Bot Zhao      1.0         None
"""

# import lib
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from typing import Optional, Union, List
import pdb

# import torch.nn.functional as F
# from torchsummary import summary
# import segmentation_models_pytorch as smp

try:
    from inplace_abn import InPlaceABN
except ImportError:
    InPlaceABN = None


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ArgMax(nn.Module):

    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim)


class Activation(nn.Module):

    def __init__(self, name, **params):

        super().__init__()

        if name is None or name == 'identity':
            self.activation = nn.Identity(**params)
        elif name == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif name == 'softmax2d':
            self.activation = nn.Softmax(dim=1, **params)
        elif name == 'softmax':
            self.activation = nn.Softmax(**params)
        elif name == 'logsoftmax':
            self.activation = nn.LogSoftmax(**params)
        elif name == 'tanh':
            self.activation = nn.Tanh()
        elif name == 'argmax':
            self.activation = ArgMax(**params)
        elif name == 'argmax2d':
            self.activation = ArgMax(dim=1, **params)
        elif callable(name):
            self.activation = name(**params)
        else:
            raise ValueError('Activation should be callable/sigmoid/softmax/logsoftmax/tanh/None; got {}'.format(name))

    def forward(self, x):
        return self.activation(x)


class Attention(nn.Module):

    def __init__(self, name, **params):
        super().__init__()

        if name is None:
            self.attention = nn.Identity(**params)
        elif name == 'scse':
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)


# class DoubleConv(nn.Module):
#     def __init__(self, inchannel, outchannel, drop_rate=0):
#         super(DoubleConv, self).__init__()
#         if drop_rate > 0:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                 nn.BatchNorm2d(outchannel),
#                 nn.Dropout2d(drop_rate),
#                 nn.LeakyReLU(),
#             )
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                 nn.BatchNorm2d(outchannel),
#                 nn.Dropout2d(drop_rate),
#                 nn.LeakyReLU(),
#             )
#
#         else:
#             self.conv1 = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                 nn.BatchNorm2d(outchannel),
#                 nn.LeakyReLU(),
#             )
#             self.conv2 = nn.Sequential(
#                 nn.Conv2d(outchannel, outchannel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                 nn.BatchNorm2d(outchannel),
#                 nn.LeakyReLU(),
#             )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         return x
#
#
# class Cls_layer(nn.Module):
#     def __init__(self):
#         super(Cls_layer, self).__init__()
#         self.max_pool = nn.AdaptiveMaxPool2d()
#         self.avg_pool = nn.AdaptiveAvgPool2d()
#
#     def forward(self, x):
#         pass
#
#
# class Unet(nn.Module):
#     def __init__(self, inchannel, n_class, drop_rate=0, filters=None):
#         super(Unet, self).__init__()
#         if not filters:
#             filters = [64, 128, 256, 512, 1024]
#         self.doubleConv1 = DoubleConv(inchannel, filters[0], drop_rate=drop_rate)
#         self.doubleConv2 = DoubleConv(filters[0], filters[1], drop_rate=drop_rate)
#         self.doubleConv3 = DoubleConv(filters[1], filters[2], drop_rate=drop_rate)
#         self.doubleConv4 = DoubleConv(filters[2], filters[3], drop_rate=drop_rate)
#         self.doubleConvBottom = DoubleConv(filters[3], filters[4], drop_rate=drop_rate)
#         self.down = nn.MaxPool2d(2)
#         self.doubleConv5 = DoubleConv(filters[4], filters[3], drop_rate=drop_rate)
#         self.doubleConv6 = DoubleConv(filters[3], filters[2], drop_rate=drop_rate)
#         self.doubleConv7 = DoubleConv(filters[2], filters[1], drop_rate=drop_rate)
#         self.doubleConv8 = DoubleConv(filters[1], filters[0], drop_rate=drop_rate)
#         self.up1 = nn.ConvTranspose2d(filters[4], filters[4] // 2, 2, stride=(2, 2))
#         self.up2 = nn.ConvTranspose2d(filters[3], filters[3] // 2, 2, stride=(2, 2))
#         self.up3 = nn.ConvTranspose2d(filters[2], filters[2] // 2, 2, stride=(2, 2))
#         self.up4 = nn.ConvTranspose2d(filters[1], filters[1] // 2, 2, stride=(2, 2))
#         self.out = nn.Sequential(
#             nn.Conv2d(filters[0], n_class, 1),
#             # nn.Softmax()
#         )
#
#     def forward(self, x):
#         x1 = self.doubleConv1(x)
#         x = self.down(x1)
#         x2 = self.doubleConv2(x)
#         x = self.down(x2)
#         x3 = self.doubleConv3(x)
#         x = self.down(x3)
#         x4 = self.doubleConv4(x)
#         x = self.down(x4)
#         x = self.doubleConvBottom(x)
#         x = self.up1(x)
#         x = self.doubleConv5(torch.cat([x4, x], 1))
#         x = self.up2(x)
#         x = self.doubleConv6(torch.cat([x3, x], 1))
#         x = self.up3(x)
#         x = self.doubleConv7(torch.cat([x2, x], 1))
#         x = self.up4(x)
#         x = self.doubleConv8(torch.cat([x1, x], 1))
#         x = self.out(x)
#         return x


class Unet_cls(nn.Module):
    def __init__(self, encoder_block, seg_class, cls_class, encoder_depth=5):
        super().__init__()
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=cls_class,  # define number of output labels
        )
        self.model = smp.Unet(encoder_name=encoder_block,
                              # choose encoder, e.g. mobilenet_v2 or efficientnet-b7, 'ResNeXt-101_swsl'
                              encoder_weights="ssl",
                              encoder_depth=encoder_depth,
                              # use `imagenet` pre-trained weights for encoder initialization
                              in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                              classes=seg_class,  # model output channels (number of classes in your dataset)
                              aux_params=aux_params)

    def forward(self, x):
        return self.model(x)

    # @staticmethod
    def get_vec(self, x):
        x = self.model.encoder(x)
        pool = nn.AdaptiveAvgPool2d(1)
        x = pool(x[-1].clone())
        return x.squeeze()


class UnetPlusPlus_Cls(nn.Module):
    def __init__(self, encoder_block, seg_class, cls_class, encoder_depth=5):
        super().__init__()
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=cls_class,  # define number of output labels
        )
        self.model = smp.UnetPlusPlus(encoder_name=encoder_block,
                                      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7, 'ResNeXt-101_swsl'
                                      encoder_weights="swsl",
                                      encoder_depth=encoder_depth,
                                      # use `imagenet` pre-trained weights for encoder initialization
                                      in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                      classes=seg_class,  # model output channels (number of classes in your dataset)
                                      aux_params=aux_params)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_vec(self, x):
        x = self.model.encoder(x)
        pool = nn.AdaptiveAvgPool2d(1)
        x = pool(x[-1].clone())
        return x.squeeze()


class ClassificationHead(nn.Sequential):

    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        self.pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        self.dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        self.linear = nn.Linear(in_channels, classes, bias=True)
        self.activation = Activation(activation)

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.shape[0], -1)
        x = self.linear(x)


class UnetPP_Cls_age(nn.Module):
    """
    age prior
    """

    def __init__(self,
                 encoder_name: str = "resnet18",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.encoder = smp.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = smp.ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.interp = nn.Sequential(
            nn.Conv2d(self.encoder.out_channels[-1], self.encoder.out_channels[-1], kernel_size=3),
            nn.LeakyReLU()
        )

        self.name = "unetplusplus-{}".format(encoder_name)
        # self.initialize()

    def forward(self, x, age, sex):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        feature_in = features[-1]
        feature_in = feature_in.permute(3, 1, 2, 0)
        feature_in = feature_in * age + sex
        feature_in = feature_in.permute(3, 1, 2, 0)
        label_final = self.classification_head(feature_in)
        # pdb.set_trace()
        return masks, label_final


class UnetPlusPlus_Cls_v2(nn.Module):
    def __init__(self, encoder_block, seg_class, cls_class, encoder_depth=5):
        super().__init__()
        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=cls_class,  # define number of output labels
        )
        self.model = smp.UnetPlusPlus(encoder_name=encoder_block,
                                      # choose encoder, e.g. mobilenet_v2 or efficientnet-b7, 'ResNeXt-101_swsl'
                                      encoder_weights="swsl",
                                      encoder_depth=encoder_depth,
                                      # use `imagenet` pre-trained weights for encoder initialization
                                      in_channels=2,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                                      classes=seg_class,  # model output channels (number of classes in your dataset)
                                      aux_params=aux_params)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def get_vec(self, x):
        x = self.model.encoder(x)
        pool = nn.AdaptiveAvgPool2d(1)
        x = pool(x[-1].clone())
        return x.squeeze()


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder
        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
        blocks[f'x_{0}_{len(self.in_channels) - 1}'] = \
            DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx + 1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] = \
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i - 1}'],
                                                                  cat_features)
        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth - 1}'])
        return dense_x[f'x_{0}_{self.depth}']


class UnetPlusPlus_Cls_V2(nn.Module):
    """
    deep supervised
    """

    def __init__(self,
                 encoder_name: str = "resnet34",
                 encoder_depth: int = 5,
                 encoder_weights: Optional[str] = "imagenet",
                 decoder_use_batchnorm: bool = True,
                 decoder_channels: List[int] = (256, 128, 64, 32, 16),
                 decoder_attention_type: Optional[str] = None,
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: Optional[Union[str, callable]] = None,
                 aux_params: Optional[dict] = None,
                 ):
        super().__init__()
        self.encoder_1 = smp.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder_2 = smp.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder_3 = smp.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        self.encoder_4 = smp.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder_1.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = smp.SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head_1 = smp.ClassificationHead(
                in_channels=self.encoder_1.out_channels[-1], **aux_params
            )
            self.classification_head_2 = smp.ClassificationHead(
                in_channels=self.encoder_2.out_channels[-1], **aux_params
            )
            self.classification_head_3 = smp.ClassificationHead(
                in_channels=self.encoder_3.out_channels[-1], **aux_params
            )
            self.classification_head_4 = smp.ClassificationHead(
                in_channels=self.encoder_4.out_channels[-1], **aux_params
            )
            self.classification_head_final = smp.ClassificationHead(
                in_channels=self.encoder_4.out_channels[-1], classes=4, pooling="avg", dropout=0.2,
            )
        else:
            self.classification_head = None

        self.interp = nn.Sequential(
            nn.Conv2d(self.encoder_4.out_channels[-1] * 4, self.encoder_4.out_channels[-1], kernel_size=3),
            nn.LeakyReLU()
        )

        self.name = "unetplusplus-{}".format(encoder_name)
        # self.initialize()

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features1 = self.encoder_1(x)
        features2 = self.encoder_2(x)
        features3 = self.encoder_3(x)
        features4 = self.encoder_4(x)
        features = []
        for i in range(len(features1)):
            temp = (features1[i] + features2[i] + features3[i] + features4[i]) / 4
            features.append(temp)

        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)

        labels_1 = self.classification_head_1(features1[-1])
        labels_2 = self.classification_head_2(features2[-1])
        labels_3 = self.classification_head_3(features3[-1])
        labels_4 = self.classification_head_4(features4[-1])
        features_in = torch.cat((features1[-1], features2[-1], features3[-1], features4[-1]), 1)
        features_in = self.interp(features_in)
        label_final = self.classification_head_final(features_in)
        return masks, label_final, labels_1, labels_2, labels_3, labels_4


if __name__ == '__main__':
    # from torchvision import transforms as tfs

    model = UnetPP_Cls_age(
        encoder_name="resnet18", in_channels=1, classes=2, aux_params=dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation=None,  # activation function, default is None
            classes=2,  # define number of output labels
        ))
    x = torch.rand(4, 1, 256, 256)
    y = model(x, torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1]))
    print(len(y))
    # import pdb
    #
    # pdb.set_trace()
# class Multitask_Unet(nn.Module):
#     def __init__(self, inchannel, seg_n_class, cls_n_class, drop_rate=0, filters=None):
#         super(Multitask_Unet, self).__init__()
#         aux_params = dict(
#             pooling='avg',  # one of 'avg', 'max'
#             dropout=drop_rate,  # dropout ratio, default is None
#             activation='softmax',  # activation function, default is None
#             classes=cls_n_class,  # define number of output labels
#         )
#         self.model = smp.Unet(
#                         encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#                         encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
#                         in_channels=inchannel,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#                         classes=seg_n_class,  # model output channels (number of classes in your dataset)
#                         decoder_use_batchnorm=True,
#                         aux_params=aux_params,
#                     )
#
#     def forward(self, x):
#         mask, label = self.model(x)
#         return mask, label

#
#
# class Unet(nn.Module):
#     def __init__(self, in_channels=1, n_classes=2,  feature_scale=2, filters=None, is_deconv=True, is_batchnorm=True):
#         super(Unet, self).__init__()
#         self.in_channels = in_channels
#         self.feature_scale = feature_scale
#         self.is_deconv = is_deconv
#         self.is_batchnorm = is_batchnorm
#
#         if not filters:
#             filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.maxpool = nn.MaxPool2d(kernel_size=2)
#         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
#         # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
#         self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
#         self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
#         # final conv (without any concat)
#         self.final = nn.Conv2d(filters[0], n_classes, 1)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)  # 16*512*512
#         maxpool1 = self.maxpool(conv1)  # 16*256*256
#
#         conv2 = self.conv2(maxpool1)  # 32*256*256
#         maxpool2 = self.maxpool(conv2)  # 32*128*128
#
#         conv3 = self.conv3(maxpool2)  # 64*128*128
#         maxpool3 = self.maxpool(conv3)  # 64*64*64
#
#         conv4 = self.conv4(maxpool3)  # 128*64*64
#         maxpool4 = self.maxpool(conv4)  # 128*32*32
#
#         center = self.center(maxpool4)  # 256*32*32
#         up4 = self.up_concat4(center, conv4)  # 128*64*64
#         up3 = self.up_concat3(up4, conv3)  # 64*128*128
#         up2 = self.up_concat2(up3, conv2)  # 32*256*256
#         up1 = self.up_concat1(up2, conv1)  # 16*512*512
#
#         final = self.final(up1)
#
#         return final
#
#
# if __name__ == '__main__':
#     import pdb
#     import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# model = Unet(1, 2, drop_rate=0.3).cuda()
# model = Multitask_Unet(inchannel=1, seg_n_class=2, cls_n_class=2).cuda()
#     ###########################################
#     model = smp.Unet(
#         encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights=None,     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=2,                      # model output channels (number of classes in your dataset)
#     ).cuda()
#     model.train()
#     # summary(model, (1, 256, 256))
#
#     x = torch.rand((8, 1, 144, 144, 144)).cuda()
#     y = model(x)
#     pdb.set_trace()
#     print(y.cpu().detach().size())
#     pdb.set_trace()
#
