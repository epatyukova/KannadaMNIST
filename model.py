#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:25:50 2022

@author: patyukoe
"""


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.modules as nn
import torch.optim as optim
import random
import csv
import pandas as pd

import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=(5,5))
        self.norm1=nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3))
        self.norm2=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3))
        self.norm3=nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(3,3)
        self.fc1 = nn.Linear(256, 100)
        self.drop=nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, num_classes)


    def forward(self, x):
        x = F.leaky_relu(self.norm1(self.conv1(x)))
        x = self.pool1(x)
        x = F.leaky_relu(self.norm2(self.conv2(x)))
        x = self.pool1(x)
        x = F.leaky_relu(self.norm3(self.conv3(x)))
        x = self.pool2(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)     
        return x
