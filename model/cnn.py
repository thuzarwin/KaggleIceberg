# -*- coding: utf-8 -*-

"""
CNN Model for Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNet(nn.Module):

    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=5, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)

        self.conv2_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(num_features=512)

        self.conv3_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc_1 = nn.Linear(8192, 512)
        self.fc_2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):

        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.5)   # (64x73x73)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.5)   # (128x73x73)
        x = F.leaky_relu(self.conv1_3(x), negative_slope=0.5)   # (256x73x73)
        x = self.bn1(x)
        x = self.pool(x)                                        # (256x36x36)
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.5)   # (512x34x34)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.5)   # (512x32x32)
        x = self.bn2(x)
        x = self.pool(x)                                        # (512x16x16)
        x = F.leaky_relu(self.conv3_1(x), negative_slope=0.5)
        x = self.bn3(x)
        x = self.pool(x)                                        # (512x8x8)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.5)   # (512x4x4)
        x = self.pool(x)                                        # (512x4x4)
        x = self.bn3(x)
        x = x.view(x.size()[0], 512*4*4)

        x = F.leaky_relu(self.fc_1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc_2(x), negative_slope=0.2)
        x = F.softmax(self.out(x))

        return x



