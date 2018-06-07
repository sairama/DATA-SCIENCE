# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 12:46:23 2018

@author: sai ram
"""

import numpy as np #mathematical tools #mathematics
import matplotlib.pyplot as plt  #plottings of the data
import pandas as pd #download and manage datasets
import os

#Importing thr dataset

os.chdir('G:\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 7 - Support Vector Regression (SVR)')

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values   #all the lines, all the columns except the last one , take all the values

y = dataset.iloc[:,2:3].values  #only the last column , index = 2

"""from sklearn.cross_validation import train_test_split

X_train , X_test,y_train, y_test = train_test_split(X,Y,test_size = 1/3,random_state =0)"""

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_x =  StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)


#FITTING  svr model

from sklearn.svm import SVR
regressor = SVR(kernel ='rbf')

regressor.fit(X,y)


#predict result from svr

y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))


#visualizing svr  model

plt.scatter(X,y,COLOR ='red')
plt.plot(X, regressor.predict(X),color = 'blue')
plt.title('truth or bluff( regression)')
plt.xlabel('position')
plt.ylabel('level')
plt.show()


