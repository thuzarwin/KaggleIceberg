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
        self.conv1_1 = nn.Conv2d(in_channels=2, out_channels=32, kernel_size=6, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)

        self.conv2_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv2_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=512)

        self.conv3_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_1 = nn.Linear(8192, 512)
        self.fc_2 = nn.Linear(512, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, x):

        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.5)
        # (16L, 32L, 72L, 72L)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.5)
        # (16L, 64L, 70L, 70L)
        x = F.leaky_relu(self.conv1_3(x), negative_slope=0.5)
        x = self.bn1(x)
        x = self.pool(x)
        # (16L, 128L, 35L, 35L)
        x = F.leaky_relu(self.conv2_1(x), negative_slope=0.5)
        # (16L, 256L, 33L, 33L)
        x = self.pool(x)
        # (16L, 256L, 16L, 16L)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.5)
        # (16L, 512L, 16L, 16L)
        x = self.bn2(x)
        x = self.pool(x)
        # (16L, 512L, 8L, 8L)
        x = F.leaky_relu(self.conv3_2(x), negative_slope=0.5)
        # (16L, 512L, 8L, 8L)
        x = self.pool(x)
        # (16L, 512L, 4L, 4L)
        x = self.bn3(x)

        x = x.view(x.size()[0], 512*4*4)

        x = F.leaky_relu(self.fc_1(x), negative_slope=0.2)
        x = F.relu(self.fc_2(x))
        x = F.softmax(self.out(x))

        return x



