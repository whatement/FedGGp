#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 15:16:37 2021

@author: fabian
"""
import os
import torch
import torchvision
import tarfile
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                  # Generate predictions        
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def get_embedding(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        _,embbedding = self(images)                    # Generate predictions                
        return embbedding
    
    def validation_step(self, batch):
        images, labels = batch 
        images = images.cuda()
        #labels = labels.cuda()
        out,_ = self(images)                    # Generate predictions        
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(),
              ]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True, pool_no=2)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 256, pool=True, pool_no=2)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))
        
        self.amp = nn.AdaptiveMaxPool2d((1, 1))
        self.FlatFeats = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)
        
        self.feature_dim = 256

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)        
        out = self.res2(out) + out        
        out = self.amp(out) # classifier(out_emb)
        f = self.FlatFeats(out)
        out = self.fc(f)
        return f, out