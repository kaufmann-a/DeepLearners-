"""
Helper modules to build a model.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import torch
import torch.nn as nn
from torch.nn import functional as F


class PPM(nn.Module):
    """
    Pyramid Pooling Module - PPM

    Based on: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
    """

    def __init__(self, in_dim, reduction_dim, bins=(1, 2, 3, 6), concat_input=True):
        super(PPM, self).__init__()
        self.concat_input = concat_input
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        if self.concat_input:
            out = [x]
        else:
            out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling - APP

    Based on https://arxiv.org/pdf/1802.02611v3.pdf,
             https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py
    """

    def __init__(self, in_dims, out_dims,
                 kernels=[1, 3, 3, 3], dilation=[1, 6, 12, 18],
                 use_bn_relu_out=False, use_global_avg_pooling=True):
        super(ASPP, self).__init__()

        self.out_dims = out_dims
        self.use_global_avg_pooling = use_global_avg_pooling

        # hardcoded blocks so that _init_weights() works (probably there is a better method)
        self.aspp_block_1 = self.get_aspp_block(in_dims, out_dims, kernels[0], 0, dilation[0])
        self.aspp_block_2 = self.get_aspp_block(in_dims, out_dims, kernels[1], dilation[1], dilation[1])
        self.aspp_block_3 = self.get_aspp_block(in_dims, out_dims, kernels[2], dilation[2], dilation[2])
        self.aspp_block_4 = self.get_aspp_block(in_dims, out_dims, kernels[3], dilation[3], dilation[3])

        nr_blocks = len(dilation)

        if self.use_global_avg_pooling:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(in_dims, out_dims,
                                                           kernel_size=(1, 1), stride=(1, 1), bias=False),
                                                 nn.BatchNorm2d(out_dims),
                                                 nn.ReLU())
            nr_blocks += 1

        if use_bn_relu_out:
            self.output_block = nn.Sequential(
                nn.Conv2d(nr_blocks * out_dims, out_dims, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_dims),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.output_block = nn.Sequential(
                nn.Conv2d(nr_blocks * out_dims, out_dims, kernel_size=(1, 1), bias=False),
                nn.Dropout(0.5)
            )

        self._init_weights()

    @staticmethod
    def get_aspp_block(in_dims, out_dims, kernel_size, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(in_dims, out_dims,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=(padding, padding),
                      dilation=(dilation, dilation),
                      bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.aspp_block_1(x)
        x2 = self.aspp_block_2(x)
        x3 = self.aspp_block_3(x)
        x4 = self.aspp_block_4(x)

        if self.use_global_avg_pooling:
            x_gap = self.global_avg_pool(x)
            x_gap = F.interpolate(x_gap, size=x1.size()[2:], mode='bilinear', align_corners=True)

            out = torch.cat([x1, x2, x3, x4, x_gap], dim=1)
        else:
            out = torch.cat([x1, x2, x3, x4], dim=1)

        return self.output_block(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
