# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 03:12:35 2020

@author: kb
"""

import torch
from torch.utils import data
import numpy as np
from scipy.io import loadmat


class ECGDataset(data.Dataset):
  
  def __init__(self, apnea_file='train_apnea.mat', normal_file='train_normal.mat', subset = 'train'):
      
      self.data_apnea = loadmat(apnea_file)[subset + '_apnea']
      self.data_normal = loadmat(normal_file)[subset + '_normal']
      
      self.lable_apnea = np.ones(self.data_apnea.shape[0])
      self.lable_normal = np.zeros(self.data_normal.shape[0])
      
      self.X = np.concatenate([self.data_apnea,self.data_normal],axis=0)
      self.X = np.asarray(self.X).astype(np.float32)
      self.X = self.X[..., np.newaxis]
      
      self.Y=np.concatenate([self.lable_apnea,self.lable_normal],axis=0)
      self.Y = np.asarray(self.Y).astype(np.float32)
      self.Y = self.Y[..., np.newaxis]
      self.Y = self.Y[..., np.newaxis]

  def __len__(self):
        return len(self.Y)

  def __getitem__(self, index):
        # Select sample
        x = self.X[index]
        y = self.Y[index]
        
        x = x.transpose(1, 0)
        y = y.transpose(1, 0)
        
        x = torch.from_numpy(np.asarray(x.astype(np.float32)))
        y = torch.from_numpy(np.asarray(y.astype(np.float32)))

        return x, y