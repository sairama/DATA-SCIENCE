# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 21:28:23 2018

@author: sai ram
"""


import numpy as np #mathematical tools #mathematics
import matplotlib.pyplot as plt  #plottings of the data
import pandas as pd #download and manage datasets
import os

#Importing thr dataset

dataset = pd.read_csv('Salary_Data.csv')

X = dataset.iloc[:,:-1].values   #all the lines, all the columns except the last one , take all the values

Y = dataset.iloc[:,1].values  #only the last column , index = 3


from sklearn.cross_validation import train_test_split

X_train , X_test,y_train, y_test = train_test_split(X,Y,test_size = 1/3,random_state =0)

"""from sklearn.preprocessing import StandardScaler

sc_x =  StandardScaler()

X_train = sc_x.fit_transform(X_train)

X_test = sc_x.fit_transform(X_test)"""

#fitting simple linear regression to the training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#Predict the test results

y_pred  = regressor.predict( X_test)

#Visualizing the train results

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict( X_train), color = 'blue')
plt.title('Salary vs experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

y_pred  = regressor.predict( X_test)

#Visualizing the test results

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict( X_train), color = 'blue')
plt.title('Salary vs experience (training set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()


