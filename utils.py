#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:27:02 2022

@author: patyukoe
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn

import random
import csv
import pandas as pd

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms



class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.set = df
        self.transform = transform

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        image = self.set.loc[idx][1:].values.astype(np.float32)
        image=torch.from_numpy(np.reshape(image,(1,28,28)))
        label = self.set.loc[idx][0]
        if self.transform:
            image = self.transform(image)

        return image, label
    
class TestDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.set = df
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        image = self.set.loc[idx][1:].values.astype(np.float32)
        image=np.reshape(image,(1,28,28))
        ids = self.set.loc[idx][0]
        if self.transform:
            image = self.transform(image)
        
        return image, ids
    
def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
             layer.reset_parameters()
             
            
def test_func(dataloader, model, device, loss_fn, mode='test'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.to(device)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if(mode =='test'):
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    elif(mode =='train'):
        print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct, test_loss