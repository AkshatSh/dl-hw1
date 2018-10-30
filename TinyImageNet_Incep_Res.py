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
    def __init__(self, in_channels, out_channels, device):
        super(Block, self).__init__()
        self.device = device
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lrelu = nn.LeakyReLU()
        self.convweight = torch.autograd.Variable(torch.randn(3).to(device), requires_grad=True)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        residual = x
        batch_size = x.shape[0]
        residual_channels = x.shape[1]
        
        
        # prob_weight = F.softmax(self.convweight)
        
        conv4out = (self.lrelu(self.conv4(x)))
        conv3out = (self.lrelu(self.conv3(x))) # * prob_weight[2]
        conv2out = (self.lrelu(self.conv2(x))) # * prob_weight[1]
        conv1out = (self.lrelu(self.conv1(x))) # * prob_weight[0]

        convout = self.bn(self.lrelu(conv4out + conv3out + conv2out + conv1out))
        
        out_channels = convout.shape[1]
        output = torch.cat([
            residual, 
            torch.zeros(
                (batch_size, 
                 out_channels - residual_channels, 
                 convout.shape[2], 
                 convout.shape[3]
                )
            ).to(self.device)
        ], dim=1)
        
        output = output + convout
        return F.max_pool2d(convout, 2)


class TinyImagenetNet(nn.Module):
    def __init__(self, device):
        super(TinyImagenetNet, self).__init__()
        # TODO define the layers
        self.conv1 = Block(3, 10, device)
        self.conv2 = Block(10, 20, device)
        self.conv3 = Block(20, 40, device)
        self.conv4 = Block(40, 80, device)
        self.conv5 = Block(80, 160, device)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(640, 320)
        self.fc2 = nn.Linear(320, 200) # 200 classes
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
        x = F.relu(self.fc1(x))
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