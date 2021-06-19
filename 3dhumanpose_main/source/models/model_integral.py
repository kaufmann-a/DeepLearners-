"""
Integral Human Pose Model
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Kouroche Bouchiat'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, kbouchiat@student.ethz.ch"

import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torchvision.models.resnet import BasicBlock, Bottleneck

from source.logcreator.logcreator import Logcreator
from source.models.basemodel import BaseModel
from source.models.bottelneck_transformer_pytorch import BottleStack
from source.models.modules import PPM, ASPP


class ResNetCustom(torch.nn.Module):
    """
    The ResNetCustom code is based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, type, num_classes=1000):

        self.resnet = {'resnet18': {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
                       'resnet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
                       'resnet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
                       'resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
                       'resnet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]}}[type]

        self.inplanes = 64
        super(ResNetCustom, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.resnet['block'], 64, self.resnet['layers'][0])
        self.layer2 = self._make_layer(self.resnet['block'], 128, self.resnet['layers'][1], stride=2)
        self.layer3 = self._make_layer(self.resnet['block'], 256, self.resnet['layers'][2], stride=2)
        self.layer4 = self._make_layer(self.resnet['block'], 512, self.resnet['layers'][3], stride=2)
        self.avgpool = torch.nn.AvgPool2d(7)
        self.fc = torch.nn.Linear(512 * self.resnet['block'].expansion, num_classes)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class JointHeatmapDeconv2x(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        assert kernel_size in (2, 3, 4), 'kernel_size must be 2, 3 or 4'

        if kernel_size == 2:
            padding, output_padding = 0, 0
        elif kernel_size == 3:
            padding, output_padding = 1, 1
        elif kernel_size == 4:
            padding, output_padding = 1, 0

        super().__init__(
            torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding,
                                     output_padding=output_padding, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )


# Alternative to Deconv2x layer but only supports 3x3 convolution
class JointHeatmapUpsample2x(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            torch.nn.Upsample(scale_factor=2, mode='bilinear'),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )


class JointHeatmapDecoder(torch.nn.Module):
    def __init__(self, num_in_channels, num_layers, num_filters, kernel_size, num_joints, depth_dim, upsample2x=False):
        super().__init__()

        self.in_channels = num_in_channels
        self.num_joints = num_joints
        self.depth_dim = depth_dim

        upsample_module_list = []
        for i in range(num_layers):
            in_channels = self.in_channels if i == 0 else num_filters
            if upsample2x:
                upsample_module_list.append(
                    JointHeatmapUpsample2x(in_channels=in_channels, out_channels=num_filters))
            else:
                # transposed convolution
                upsample_module_list.append(
                    JointHeatmapDeconv2x(in_channels=in_channels, out_channels=num_filters, kernel_size=kernel_size))

        self.upsample_features = torch.nn.Sequential(*upsample_module_list)

        self.features_to_heatmaps = torch.nn.Conv2d(num_filters, num_joints * depth_dim, kernel_size=1)

        JointHeatmapDecoder._init_weights(self)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0, std=0.001)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, torch.nn.ConvTranspose2d):
            torch.nn.init.normal_(module.weight, mean=0, std=0.001)

        elif isinstance(module, torch.nn.Module):
            for m in module.children():
                JointHeatmapDecoder._init_weights(m)

    def forward(self, features):
        features = self.upsample_features(features)
        heatmaps = self.features_to_heatmaps(features)

        N, _, H, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, self.num_joints, self.depth_dim, H, W)
        return heatmaps  # Return format N, J, D, H, W


class JointIntegralRegressor(torch.nn.Module):
    def __init__(self, infer_with_argmax=False):
        super().__init__()
        self.infer_with_argmax = infer_with_argmax

    def forward(self, heatmaps):
        N, J, D, H, W = heatmaps.shape

        # Apply global softmax to the heatmaps
        heatmaps = heatmaps.reshape(N, J, D * H * W)
        heatmaps = F.softmax(heatmaps, dim=2)
        heatmaps = heatmaps.reshape(N, J, D, H, W)

        # Integrate over other axes
        x_maps = heatmaps.sum(dim=2).sum(dim=2)
        y_maps = heatmaps.sum(dim=2).sum(dim=3)
        z_maps = heatmaps.sum(dim=3).sum(dim=3)

        if not self.training and self.infer_with_argmax:
            return torch.stack((
                x_maps.argmax(dim=2) / W - 0.5,
                y_maps.argmax(dim=2) / H - 0.5,
                z_maps.argmax(dim=2) / D - 0.5
            ), dim=2)

        # Take expected coordinate for each axis (and recenter)
        x_preds = (x_maps * torch.arange(W).to(heatmaps.device)).sum(dim=2) / W - 0.5
        y_preds = (y_maps * torch.arange(H).to(heatmaps.device)).sum(dim=2) / H - 0.5
        z_preds = (z_maps * torch.arange(D).to(heatmaps.device)).sum(dim=2) / D - 0.5

        return torch.stack((x_preds, y_preds, z_preds), dim=2)  # Return format N, J, 3


class ModelIntegralPoseRegression(BaseModel):
    """
    Based on https://github.com/JimmySuen/integral-human-pose
    """

    name = 'IntegralPoseRegressionModel'
    resnet_nr_output_channels = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
        "resnet101": 2048,
        "resnet152": 2048
    }

    def __init__(self, model_params, dataset_params):
        super().__init__()

        self.backbone = self.getPretrainedResnet(model_params, pretrained=True)

        num_in_channels = self.resnet_nr_output_channels[model_params.resnet_model]

        self.train_heatmaps_only = hasattr(model_params, 'train_heatmaps_only') and model_params.train_heatmaps_only

        if hasattr(model_params, "use_bot_net") and model_params.use_bot_net:
            fmap_size = [v // (2 ** 4) for v in dataset_params.image_size]  # might not work for all sizes
            fmap_size = fmap_size[::-1]  # reverse to get H x W
            fmap_size = tuple(fmap_size)
            layer = BottleStack(
                dim=num_in_channels // 2,
                fmap_size=fmap_size,
                dim_out=num_in_channels,
                proj_factor=4,
                num_layers=3,
                downsample=True,  # downsample on first layer
                heads=4,
                dim_head=128,
                rel_pos_emb=True,  # use relative positional embedding
                activation=nn.ReLU()
            )

            resnet_layers = list(self.backbone.children())

            self.backbone = nn.Sequential(
                *resnet_layers[:7],
                layer
            )

        self.bottleneck = None
        if hasattr(model_params, "bottleneck"):
            if model_params.bottleneck.method == "PPM":
                bins = model_params.bottleneck.bins
                concat_input = True
                reduction_dim = num_in_channels // len(bins)
                out_dim = reduction_dim * len(bins) + (num_in_channels if concat_input else 0)

                self.bottleneck = PPM(in_dim=num_in_channels,
                                      reduction_dim=reduction_dim,
                                      bins=bins,
                                      concat_input=concat_input)
                num_in_channels = out_dim

            elif model_params.bottleneck.method == "ASPP":
                out_dim = int(num_in_channels * model_params.bottleneck.out_dim_factor)

                self.bottleneck = ASPP(in_dims=num_in_channels,
                                       out_dims=out_dim,
                                       dilation=model_params.bottleneck.dilation,
                                       use_bn_relu_out=True,
                                       use_global_avg_pooling=model_params.bottleneck.use_global_avg_pooling)
                num_in_channels = out_dim

        self.joint_decoder = JointHeatmapDecoder(num_in_channels=num_in_channels,
                                                 num_layers=model_params.num_deconv_layers,
                                                 num_filters=model_params.num_deconv_filters,
                                                 kernel_size=model_params.kernel_size,
                                                 num_joints=model_params.num_joints,
                                                 depth_dim=model_params.depth_dim,
                                                 upsample2x=model_params.deconv_use_upsample if hasattr(model_params,
                                                                                                        "deconv_use_upsample") else False
                                                 )
        self.joint_regressor = JointIntegralRegressor(
            infer_with_argmax=model_params.infer_with_argmax if hasattr(model_params, "infer_with_argmax") else False)

        Logcreator.info("Successfully initialized model with name IntegralPoseRegressionModel")

    def getPretrainedResnet(self, model_params, pretrained=True):
        model = ResNetCustom(model_params.resnet_model)

        model_urls = {
            'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
            'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
            'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
            'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
        }

        if pretrained:
            model.load_state_dict(model_zoo.load_url(model_urls[model_params.resnet_model]))
        return model

    def forward(self, input):
        features = self.backbone(input)
        if self.bottleneck is not None:
            features = self.bottleneck(features)
        heatmaps = self.joint_decoder(features)

        if not self.train_heatmaps_only or not self.training:
            joints = self.joint_regressor(heatmaps)
        else:
            joints = None

        return heatmaps, joints
