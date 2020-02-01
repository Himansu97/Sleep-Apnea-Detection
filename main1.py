# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 11:39:58 2020

@author: asus
"""

import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, Conv1D, GlobalAveragePooling1D, MaxPooling1D
from keras import regularizers
from keras.callbacks import ModelCheckpoint,EarlyStopping
import time
filename="ApneaData.pkl"
testpercent=20
features=[]
classes=[]
t=time.time()
f=open(filename,'rb')
data=pickle.load(f)
f.close()
np.random.shuffle(data)
for row in data:
    features.append(row[:-1])
    classes.append(row[-1])
number_of_classes=2
inputLength=len(features)
testLength=int(inputLength*0.2)
big=len(features[0])
X_train,Y_train=features[:-testLength],classes[:-testLength]
X_test,Y_test=features[-testLength:],classes[-testLength:]
print("Processing time: ",(time.time()-t))
t=time.time()

###############################
X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
Y_test = Y_test.reshape(Y_test.shape[0], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

###############################
model=Sequential()
model.add(Conv1D(20, 60, activation="relu",input_shape=(big, 1)))
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
checkpointer=ModelCheckpoint(filepath="C:/Users/asus/Desktop/Hapi/Sleep apnea/BestModel.h5",monitor='val_acc',verbose=1,save_best_only='True')
es=EarlyStopping(monitor='val_acc')
#es=[checkpointer,cb]
hist=model.fit(X_train,Y_train,validation_data=(X_test,Y_test),batch_size=100,epochs=30, verbose=2,callbacks=[es])
predictions=model.predict(X_test)
score = accuracy_score(Y_test,predictions)
model.save("Best_model.h5")