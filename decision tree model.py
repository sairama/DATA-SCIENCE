# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:04:07 2018

@author: sai ram
"""

import numpy as np #mathematical tools #mathematics
import matplotlib.pyplot as plt  #plottings of the data
import pandas as pd #download and manage datasets
import os

#Importing thr dataset

os.chdir('G:\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 8 - Decision Tree Regression')

dataset = pd.read_csv('Position_Salaries.csv')

X = dataset.iloc[:,1:2].values   #all the lines, all the columns except the last one , take all the values

Y = dataset.iloc[:,2].values  #only the last column , index = 2

"""from sklearn.cross_validation import train_test_split

X_train , X_test,y_train, y_test = train_test_split(X,Y,test_size = 1/3,random_state =0)"""



#FITTING   decsion tree REGRESSION model to the dataset

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state =0)
regressor.fit(X,Y)




#predict result from polynomial regressin

y_pred = regressor.predict(6.5)

#visualizing polynomial regression model

#visualizing  regression model for smoother curve
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y,COLOR ='red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('truth or bluff( decision tree regression)')
plt.xlabel('position')
plt.ylabel('level')
plt.show()


