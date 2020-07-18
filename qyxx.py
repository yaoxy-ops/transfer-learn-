# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:43:08 2020

@author: Lenovo
"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
import matplotlib.pyplot as plt
from pylab import mpl 
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
import keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import load_model
from math import sqrt
from sklearn.metrics import r2_score,mean_squared_error

inputa = pd.read_excel(r'C:/Users/Lenovo/Desktop/MATLAB/迁移学习/学习模型.xlsx',sheet_name="input")
outputa =pd.read_excel(r'C:/Users/Lenovo/Desktop/MATLAB/迁移学习/学习模型.xlsx',sheet_name="output")
index= np.arange(10)##用10个样品训练。

index2= np.arange(53)##测试集。这里的训练集与预测集的样品与预训练模型不是同一批
index3=np.delete(index2,index)
inputal=inputa.values
outputall=outputa.values
min_max_scaler = preprocessing.MinMaxScaler()
inputall = min_max_scaler.fit_transform(inputal)

train_x_data = inputall[index,:]#index,:
train_y_data= outputall[index,:]

test_x_data =inputall[index3,:]#index2,:
test_y_data=outputall[index3,:]
pre_base = load_model('my_model.h5')
pre_base.trainble = False
##pre_base.summary() 显示模型结构
model = tf.keras.Sequential()
model.add(pre_base)
model.add(layers.Dense(400,activation='relu'))
model.add(layers.Dense(1,activation='relu')) 
model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
model.fit(train_x_data,train_y_data,batch_size=20,epochs=400)
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] 
mpl.rcParams['axes.unicode_minus'] = False
plt.xlabel('reference( 10^-5)')
plt.ylabel('predict( 10^-5)')
plt.scatter(train_y_data,model.predict(train_x_data),c='blue',label='pre_set')
plt.scatter(test_y_data,model.predict(test_x_data),c='red',label='val_set')
plt.legend(loc = 'upper left')
plt.title('迁移模型')
r2=r2_score(test_y_data,model.predict(test_x_data))
msep=sqrt(mean_squared_error(train_y_data,model.predict(train_x_data)))
rmsep=sqrt(mean_squared_error(test_y_data,model.predict(test_x_data)))
a = np.linspace(0,5000, 5000)
b =a
plt.plot(a,b,c='red')
plt.text(3200,500,r'$R^2=%.4f$'%r2,fontsize=10,verticalalignment="bottom",horizontalalignment="left")
plt.text(3200,1000,r'$MSEP=%.4f$'%msep,fontsize=10,verticalalignment="bottom",horizontalalignment="left")
plt.text(3200,1500,r'$RMSEP=%.4f$'%rmsep,fontsize=10,verticalalignment="bottom",horizontalalignment="left")
plt.savefig('2.png',dpi=1080)
plt.savefig('2._副本.png',dpi=600)
plt.show()
