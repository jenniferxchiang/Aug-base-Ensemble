#--------------------------- edited ---------------------------#
#--------------------------- edited ---------------------------#


### Adapted for 1D data from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py ###


import numpy as np
import os
import torch
import torch.nn as nn
import torch.utils.data as utils
import torch.nn.functional as F
from torch.distributions.normal import Normal


def conv15x1(in_channels, out_channels, stride=1):
    # 15x1 convolution with padding"
    return nn.Conv1d(in_channels, out_channels, kernel_size=15, stride=stride, padding=7)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        norm_layer = nn.BatchNorm1d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv15x1(in_channels, out_channels, stride)
        self.bn1 = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)  # inplace = true, do cover computing
        self.conv2 = conv15x1(out_channels, out_channels)
        self.bn2 = norm_layer(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = x.float()
        identity = x
        # If it is the first BasicBlock in layer2, layer3, layer4, the first convolutional layer will downsample
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_outputs=1, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,):

        super(ResNet, self).__init__()

        self.num_outputs = 1
        self._norm_layer = nn.BatchNorm1d
        self.inplanes = 32

        # conv1
        self.conv1 = nn.Conv1d(
            12, self.inplanes, kernel_size=15, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # conv2
        self.layer1 = self._make_layer(
            block, 32, layers[0])  # block means one type
        # conv3
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        # conv4
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        # conv5
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, num_outputs))

        # Used to initialize each module in the network
        # nn.modules() returns all modules in the network
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #    for m in self.modules():
        #        if isinstance(m, Bottleneck):
        #            nn.init.constant_(m.bn3.weight, 0)
        #        elif isinstance(m, BasicBlock):
        #            nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # blocks represents the number of operations to be repeated by the module

        norm_layer = nn.BatchNorm1d
        downsample = None

        # if stride == 1; num of input channel == num of output channel
        # Input channel != Output channel*4, input channel is 64
        # That is to say, as long as the make_layer function is called, the downsample must be executed, but the execution occurs after the block operation
        # The first operation of each repeated convolution block must be downsampled on the bypass connection
        # Then, the remaining few operations of the convolution block will no longer perform downsampling operations
        # that is, no downsampling operations will be performed on the bypass connection

        if stride != 1 or self.inplanes != planes * block.expansion:
            # print("Got to downsample place")
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                norm_layer(planes),
            )

        layers = []
        # add first basic block, send downsample to basicblock as downsample layer
        layers.append(block(self.inplanes, planes, stride, downsample))
        # edit output channal number
        self.inplanes = planes * block.expansion
        # Continue to add the next BasicBlock in this layer
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        # list array, transform with *, split layers into elements
        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = x.to(torch.float32)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        op = self._forward_impl(x)
        return op


def resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=False, progress=False,
                  **kwargs)
