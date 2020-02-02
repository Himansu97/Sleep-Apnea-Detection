# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 02:31:30 2020

@author: kb
"""

from collections import OrderedDict

import torch
import torch.nn as nn


class ECGNet(nn.Module):

    def __init__(self, in_channels=6000, out_channels=1, init_features=64):
        super(ECGNet, self).__init__()

        features = init_features
        
        self.Block1 = ECGNet.fcn_block(in_channels, features, name="fcn1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Block2 = ECGNet.fcn_block(features, features * 2, name="fcn2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.Block3 = ECGNet.fcn_block(features * 2, features * 4, name="fcn3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.full = ECGNet.classifier_block(in_channels=features * 4, out_channels=out_channels, name='classifier')
        

    def forward(self, x):
        
        x = self.Block1(x)
        #x = self.pool1(x)
        x = self.Block2(x)
        #x = self.pool2(x)
        x = self.Block3(x)
        #x = self.pool3(x)
        
        x = x.view((-1, 256))
        x = self.full(x)
        
        return x


    @staticmethod
    def fcn_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu1", nn.ELU(inplace=True)),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            bias=False,
                        ),
                    ),
                    (name + "relu2", nn.ELU(inplace=True)),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                ]
            )
        )
            
    @staticmethod
    def classifier_block(in_channels, out_channels, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (name + "fc", nn.Linear(in_features=in_channels, out_features=out_channels, bias=False)),
                    (name + "relu", nn.ELU(inplace=True)),
                    (name + "norm", nn.BatchNorm1d(num_features=out_channels)),
                    (name + "drop", nn.Dropout(p=0.6)),
                    (name + "softmax", nn.Softmax(out_channels))
                ]
            )
        )