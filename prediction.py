# -*- coding: utf-8 -*-
"""
Created on Wed Apr 2020

@author: Ali
"""

#Predicting thermal conducitivty for 4 sands 

import pandas as pd
from pandas import DataFrame 
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Caliberation_Data.csv")
X = dataset.iloc[:, 1:3]
Y = dataset.iloc[:, 3]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import sklearn.metrics 

model = Sequential()
model.add(Dense(6, input_dim=2, activation='tanh', kernel_initializer='he_uniform'))
model.add(Dense(6, activation = 'tanh', kernel_initializer = 'he_uniform'))
model.add(Dense(6, activation = 'tanh', kernel_initializer = 'he_uniform'))
model.add(Dense(1,activation = 'linear', kernel_initializer = 'he_uniform'))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'mse', 'mape' ])
history = model.fit(X_train, Y_train,  validation_data = (X_test, Y_test), batch_size = 10 ,
                    epochs = 650)

#Getting the scores and error for the training set
Y_pred_train = model.predict(X_train)
r2_train = r2_score(Y_train, Y_pred_train) * 100
print ("R2 score: ", r2_train)
mse_train = sklearn.metrics.mean_squared_error(Y_train, Y_pred_train)
print("Mean Squared Error: ", mse_train)
mae_train = sklearn.metrics.mean_absolute_error(Y_train, Y_pred_train)
print("Mean Absolute Error: ", mae_train)


#test data set
Y_pred_test = model.predict(X_test)
r2_test = r2_score(Y_test, Y_pred_test) * 100
print ("R2 score: ", r2_test)
mse_test = sklearn.metrics.mean_squared_error(Y_test, Y_pred_test)
print("Mean Squared Error: ", mse_test)
mae_test = sklearn.metrics.mean_absolute_error(Y_test, Y_pred_test)
print("Mean Absolute Error: ", mae_test)


#importing validation data  
val_data = pd.read_csv("Validation_Data.csv")
X_val = val_data.iloc[:, 1:3]    
Y_val = val_data.iloc[:, 3]

X_val = sc.fit_transform(X_val)

Y_val_pred = model.predict(X_val)

r2_val = r2_score(Y_val, Y_val_pred) * 100
print ("R2 score: ", r2_val)
mse_val = sklearn.metrics.mean_squared_error(Y_val, Y_val_pred)
print("Mean Squared Error: ", mse_val)
mae_val = sklearn.metrics.mean_absolute_error(Y_val, Y_val_pred)
print("Mean Absolute Error: ", mae_val)

plt.plot(range(1,len(Y_val)+1), Y_val, label = 'Data' ) 
plt.plot(range(1,len(Y_val_pred)+1), Y_val_pred, label = 'Predicted' ) 
plt.legend(['Original Data', 'Predicted Data'])
plt.xlabel('Inputs')
plt.ylabel('Vp')
plt.show()