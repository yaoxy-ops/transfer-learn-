# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:55:50 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
from pylab import mpl 
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
import keras
from tensorflow.keras.optimizers import Adam, SGD
from math import sqrt

##预训练模型
inputa = pd.read_excel(r'C:/Users/Lenovo/Desktop/MATLAB/迁移学习/预训练模型.xlsx',sheet_name="input")
outputa =pd.read_excel(r'C:/Users/Lenovo/Desktop/MATLAB/迁移学习/预训练模型.xlsx',sheet_name="output")
index = np.array([46,12,55,34,6,14,16,17,8,4])
index3=np.arange(66)
index2=np.delete(index3,index)
inputal=inputa.values
outputall=outputa.values

min_max_scaler = preprocessing.MinMaxScaler()
inputall = min_max_scaler.fit_transform(inputal)

train_x_data = inputall[index2,:]
train_y_data= outputall[index2,:]

test_x_data =inputall[index,:]
test_y_data=outputall[index,:]
          #归一化

  ##

model =tf.keras.Sequential([
    Dense(40,activation='relu',input_dim=11),
    Dense(300,activation='relu'),

    Dense(1,activation='relu'),
    ])
model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])#Adam

history= model.fit(train_x_data,train_y_data,batch_size=20,epochs=805)

model.save('my_model_2.h5') 

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False
fig1=plt.figure
plt.xlabel('reference( 10^-5)')
plt.ylabel('predict( 10^-5)')
plt.scatter(train_y_data,model.predict(train_x_data),label='pre_set')
plt.scatter(test_y_data,model.predict(test_x_data),label='val_set')
plt.legend(loc = 'upper left')
plt.title('预训练模型')
r2=r2_score(train_y_data,model.predict(train_x_data))
msep=sqrt(mean_squared_error(train_y_data,model.predict(train_x_data)))
rmsep=sqrt(mean_squared_error(test_y_data,model.predict(test_x_data)))
a = np.linspace(0,10000, 10000)
b =a
plt.plot(a,b,c='red')
plt.text(7000,2000,r'$R^2=%.4f$'%r2,fontsize=10,verticalalignment="bottom",horizontalalignment="left")
plt.text(7000,3000,r'$MSEP=%.4f$'%msep,fontsize=10,verticalalignment="bottom",horizontalalignment="left")
plt.text(7000,4000,r'$RMSEP=%.4f$'%rmsep,fontsize=10,verticalalignment="bottom",horizontalalignment="left")

plt.savefig('1._副本.png',dpi=600)
plt.show()

