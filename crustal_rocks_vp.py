# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 2019

@author: Ali
"""

#Predicting Vp

#Python program to predict Vp for sand 
import pandas as pd
from pandas import DataFrame #for saving values to excel file
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Data_Caliberation.csv")
X = dataset.iloc[:, 1:8]    #seven Features
Y = dataset.iloc[:, 8]   #Vp target values


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
#70% Training data and 30% testing data

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
import sklearn.metrics 
import append_df_to_excel #to append result(presicted) into the excel files. Execute the file 'append_df_to_excel.py'
                            #before using it as a header file 

#Loop for runnin the training 20 times and saving the predicted data from the validation set into excel
#file. The same network is to be used when training and finding the P- wave velocities.

for i in range (20):
    model = Sequential()
    model.add(Dense(8, input_dim=7, activation='tanh', kernel_initializer='he_uniform'))
    model.add(Dense(8, activation = 'tanh', kernel_initializer = 'he_uniform'))
    model.add(Dense(8, activation = 'tanh', kernel_initializer = 'he_uniform'))
    model.add(Dense(1,activation = 'linear', kernel_initializer = 'he_uniform'))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae', 'mse', 'mape' ])
    history = model.fit(X_train, Y_train,  validation_data = (X_test, Y_test), batch_size = 10 ,
                    epochs = 800)

    val_data = pd.read_csv("New_Val_Data.csv")
    X_val = val_data.iloc[:, 1:8]       
    X_val = sc.fit_transform(X_val)
    Y_val_pred = model.predict(X_val)
    #Y_val_pred = [[100], [200], [300]]
    Y_val_pred = DataFrame(Y_val_pred) 
    #each iteration will save the results in each sheet in the excel
    append_df_to_excel('output.xlsx', Y_val_pred, sheet_name='Sheet'+str(i), index=False, header = False, startrow=0)



Y_val_pred.to_csv('output.csv', sheetname = 'sheet1', index=False, header = False)
    


#For getting output for single new input
new_prediction = model.predict(sc.transform(np.array([[61.2,	17.4,	6.13,	3.24,	1.28,	20,	400]])))

#Visualizing training; to get both the curves add 'validation_data =(x_test y_test)' in the fit function
import statistics as stats
loss = stats.mean(history.history['loss'])
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['mean_squared_error'])
plt.plot(history.history['val_mean_squared_error'])
plt.legend(['train', 'test'])
plt.ylabel('Mean Square Error')
plt.xlabel('Epochs')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'test'])
plt.xlabel('Epochs')
plt.ylabel('Loss')

#Getting the scores and erro for the training set
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
val_data = pd.read_csv("New_Val_Data.csv")
X_val = val_data.iloc[:, 1:8]    
Y_val = val_data.iloc[:, 8]

X_val = sc.fit_transform(X_val)

Y_val_pred = model.predict(X_val)

r2_val = r2_score(Y_val, Y_val_pred) * 100
print ("R2 score: ", r2_val)
mse_val = sklearn.metrics.mean_squared_error(Y_val, Y_val_pred)
print("Mean Squared Error: ", mse_val)
mae_val = sklearn.metrics.mean_absolute_error(Y_val, Y_val_pred)
print("Mean Absolute Error: ", mae_val)



#Visualization of validation data. This will give correct graph for result as the testing and training data are randomly chosen
#so p;otting the graph for them wont be possible. Predicting validation data and its not randomized so we gget the perfect deviation 
#graph
plt.plot(range(1,len(Y_val)+1), Y_val, label = 'Data' ) 
plt.plot(range(1,len(Y_val_pred)+1), Y_val_pred, label = 'Predicted' ) 
plt.legend(['Original Data', 'Predicted Data'])
plt.xlabel('Inputs')
plt.ylabel('Vp')
plt.show()



















