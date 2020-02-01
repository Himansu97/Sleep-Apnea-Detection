# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 23:44:04 2020

@author: asus
"""
import numpy as np
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import time
path="C:/Users/asus/Desktop/Hapi/Sleep apnea/Dataset"
train_apnea=loadmat('C:/Users/asus/Desktop/Hapi/Sleep apnea/Dataset/train_apnea.mat')
train_apnea=train_apnea['train_apnea']
train_normal=loadmat('C:/Users/asus/Desktop/Hapi/Sleep apnea/Dataset/train_normal.mat')
train_normal=train_normal['train_normal']
X_train=np.concatenate([train_apnea,train_normal],axis=0)
Y_train=np.concatenate([np.ones(6503),np.zeros(10406)])
Y_train=Y_train.reshape(Y_train.shape[0],-1)
test_apnea=loadmat('C:/Users/asus/Desktop/Hapi/Sleep apnea/Dataset/test_apnea.mat')
test_apnea=test_apnea['test_apnea']
test_normal=loadmat('C:/Users/asus/Desktop/Hapi/Sleep apnea/Dataset/test_normal.mat')
test_normal=test_normal['test_normal']
X_test=np.concatenate([test_apnea,test_normal],axis=0)
Y_test=np.concatenate([np.ones(6547),np.zeros(10697)])
Y_test=Y_test.reshape(Y_test.shape[0],-1)

########################
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],1)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(Y_test.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],1)

#####################
model=Sequential()
model.add(Conv1D(20, 60, activation="relu",input_shape=(6000, 1)))
model.add(MaxPooling1D(pool_size = 4))
model.add(Conv1D(28, 30 ,activation="relu"))
model.add(MaxPooling1D(pool_size = 2))
model.add(Conv1D(15,20,activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(2058,kernel_initializer="normal",activation="relu"))
model.add(Dense(1029,kernel_initializer="normal",activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
es=EarlyStopping(monitor='val_acc',patience=6)
hist=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=100,epochs=50,verbose=2,callbacks=[es])
predictions=model.predict(X_test)
score=accuracy_score(Y_test,predictions)

#######################

model=Sequential()
model.add(Conv1D(50, 20, activation='relu', input_shape=(6000, 1)))
model.add(MaxPooling1D(2))
model.add(Conv1D(50, 20, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(30, 24, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(30, 24, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Conv1D(10, 24, activation='relu'))
model.add(MaxPooling1D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(192,kernel_initializer='normal',activation='relu'))
model.add(Dense(192,kernel_initializer='normal',activation='relu'))
model.add(Dense(1,kernel_initializer='normal',activation='sigmoid'))
model.compile(loss="binary_crossentropy",optimizer="adam",metrics=['accuracy'])
es=EarlyStopping(monitor='val_acc',patience=6)
hist=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=100,epochs=30, verbose=2,callbacks=[es])
model.save('paper2.h5')
