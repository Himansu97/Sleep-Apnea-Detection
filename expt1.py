# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 02:08:50 2020

@author: kb
"""

import numpy as np
from scipy.io import loadmat
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

######################## training Data Prep ################################
train_apnea = loadmat('train_apnea.mat')['train_apnea'][:6500]
y_apnea = np.ones(train_apnea.shape[0])

train_normal = loadmat('train_normal.mat')['train_normal'][:10404]
y_normal = np.zeros(train_normal.shape[0])

X_train=np.concatenate([train_apnea,train_normal],axis=0)
X_train = np.asarray(X_train).astype(np.float32)
X_train = np.expand_dims(X_train, axis=2)

Y_train=np.concatenate([y_apnea,y_normal],axis=0)
Y_train = np.asarray(Y_train).astype(np.float32)
Y_train = np.expand_dims(Y_train, axis=1)

x = Variable(torch.Tensor(X_train), requires_grad=False)
y = Variable(torch.Tensor(Y_train), requires_grad=False)

train_data = TensorDataset(x, y)
train_loader = DataLoader(train_data, shuffle = True, batch_size=4)

'''
from torch.utils.data import Dataset, DataLoader

from data_prep import ECGDataset 

train_data = ECGDataset()
train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)

val_data = ECGDataset(apnea_file='test_apnea.mat', normal_file='test_normal.mat', subset='test')
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=True, num_workers=2)
'''


dataiter = iter(train_loader)
signal, label = dataiter.next()

######################## training #################################

from model import ECGNet

#model = ECGNet(in_channels=signal.shape[1], out_channels=label.shape[1])
model = ECGNet()
model = model.cuda()
model.train()

#out = model(signal)

import torch.optim as optim
import torch.nn as nn

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []

for epoch in range(10):  # loop over the dataset multiple times

    running_loss = 0.0
    train_loss = 0.0
    
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(),data[1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        train_loss += loss.item()
        if i % 500 == 499:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
    losses.append(train_loss)

print('Finished Training')

######################## validation Data Prep ################################
val_apnea = loadmat('test_apnea.mat')['test_apnea'][:6544]
y_apnea_val = np.ones(val_apnea.shape[0])

val_normal = loadmat('test_normal.mat')['test_normal'][:10696]
y_normal_val = np.zeros(val_normal.shape[0])

X_val=np.concatenate([val_apnea,val_normal],axis=0)
X_val = np.asarray(X_val).astype(np.float32)
X_val = np.expand_dims(X_val, axis=2)

Y_val=np.concatenate([y_apnea_val,y_normal_val],axis=0)
Y_val = np.asarray(Y_val).astype(np.float32)
Y_val = np.expand_dims(Y_val, axis=1)

x_val = Variable(torch.Tensor(X_val), requires_grad=False)
y_val = Variable(torch.Tensor(Y_val), requires_grad=False)

val_data = TensorDataset(x_val, y_val)
val_loader = DataLoader(val_data, shuffle = True, batch_size=4)

dataiter = iter(val_loader)
signal, label = dataiter.next()[0].cuda(), dataiter.next()[1].cuda()


# print signal
classes = ['normal', 'apnea']
signal
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j].type(torch.IntTensor)] for j in range(4)))




outputs = model(signal)

print('Predicted: ', ' '.join('%5s' % classes[outputs[j].type(torch.IntTensor)] for j in range(4)))

labels.size(0)
(outputs == labels).sum()


correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        signals, labels = data[0].cuda(), data[1].cuda()
        outputs = model(signals)
        total += labels.size(0)
        correct += (outputs == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(2))
class_total = list(0. for i in range(2))

with torch.no_grad():
    for data in val_loader:
        signals, labels = data[0].cuda(), data[1].cuda()
        outputs = model(signals)
        c = (outputs == labels).squeeze()
        for i in range(4):
            label = labels[i].type(torch.IntTensor)
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(2):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))



