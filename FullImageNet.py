import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import os
import torch.nn.functional as F
import torch.optim as optim
import h5py
import pt_util

class FullImagenetNet(TinyImagenetNet):
    def __init__(self, device):
        super(TinyImagenetNet, self).__init__()
        # TODO define the layers
        self.conv1 = Block(3, 24, device)
        self.conv2 = Block(24, 48, device)
        self.conv3 = Block(48, 96, device)
        self.conv4 = Block(96, 192, device)
        self.conv5 = Block(192, 384, device)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(6144, 1000) # 1000 classes
        self.best_acc = 0

    def forward(self, x):
        # TODO define the forward pass
        batch = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self.conv3_drop(x)
        x = self.conv4(x)
        x = self.conv5(x)
        self.conv3_drop(x)
        x = x.view(batch, -1)
        x = F.dropout(x, training=self.training)
        x = self.fc1(x)
        return x
