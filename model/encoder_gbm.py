# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import numpy as np


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(2,64, 3,stride=1,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )

        self.fc1 = nn.Linear(128*18*18, 8192)
        self.fc2 = nn.Linear(8192, 512)

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(-1, 128*18*18)
        x = F.leaky_relu(self.fc1(x), 0.03)
        x = F.leaky_relu(self.fc2(x), 0.03)
        return x


class Decoder(nn.Module):

    def __init__(self, input_size=512):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 8192)
        self.fc2 = nn.Linear(8192, 32*18*18)
        self.fc3 = nn.Linear(32*18*18, 75*75)
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 2, 3, stride=1, padding=1),
        )

    def forward(self, x):

        x = F.leaky_relu(self.fc1(x), 0.03)
        x = F.leaky_relu(self.fc2(x), 0.03)
        x = F.leaky_relu(self.fc3(x), 0.03)
        x = x.view(-1, 1, 75, 75)
        x = self.convnet(x)
        return x


def test_encoder():

    x = np.random.randn(75*75*3*2)
    x = x.reshape((3,2,75,75))
    encoder = Encoder()
    x = Variable(torch.FloatTensor(x))
    encoder(x)


def test_decoder():

    x = np.random.randn(512*3)
    x = x.reshape((3,512))
    decoder = Decoder(512)
    print(x.shape)
    x = Variable(torch.FloatTensor(x))
    print(x.size())
    decoder(x)

