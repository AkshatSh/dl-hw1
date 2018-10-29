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

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, dial)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x

        conv3out = self.bn(F.relu(self.conv3(x)))
        conv2out = self.bn(F.relu(self.conv2(x)))
        conv1out = self.bn(F.relu(self.conv1(x)))

        convout = F.relu(conv3out + conv2out + conv1out)

        return F.max_pool2d(convout + residual, 2)


class TinyImagenetNet(nn.Module):
    def __init__(self):
        super(TinyImagenetNet, self).__init__()
        # TODO define the layers
        self.conv1 = Block(3, 10)
        self.conv2 = Block(10, 20)
        self.conv3 = Block(20, 40)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(3380, 50)
        self.fc2 = nn.Linear(50, 200) # there are 200 classes
        self.best_acc = 0

    def forward(self, x):
        # TODO define the forward pass
        batch = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        self.conv3_drop(x)
        x = x.view(batch, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def loss(self, prediction, label, reduction='elementwise_mean'):
        loss_val = F.cross_entropy(prediction, label.squeeze(), reduction=reduction)
        return loss_val

    def save_model(self, file_path, num_to_keep=1):
        pt_util.save(self, file_path, num_to_keep)
        
    def save_best_model(self, accuracy, file_path, num_to_keep=1):
        # TODO save the model if it is the best
        if self.best_acc < accuracy:
          self.best_acc = accuracy
          self.save_model(file_path, num_to_keep)

    def load_model(self, file_path):
        pt_util.restore(self, file_path)

    def load_last_model(self, dir_path):
        return pt_util.restore_latest(self, dir_path)

