# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:56:32 2018

@author: sai ram
"""

import numpy as np #mathematical tools #mathematics
import matplotlib.pyplot as plt  #plottings of the data
import pandas as pd #download and manage datasets
import os

#Importing thr dataset

os.chdir('G:\\Machine_Learning_AZ_Template_Folder\Machine Learning A-Z Template Folder\\Part 2 - Regression\\Section 5 - Multiple Linear Regression')

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[:,:-1].values   #all the lines, all the columns except the last one , take all the values
y = dataset.iloc[:,4].values  #only the last column , index = 3


#ENCODING CATEGORICAL variable

from sklearn.preprocessing import  LabelEncoder ,OneHotEncoder

labelencoder_X = LabelEncoder()
x[:,3]= labelencoder_X.fit_transform(x[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])

x = onehotencoder.fit_transform(x).toarray()

#Avoiding the dummy variable trap

x = x[:,1:]


from sklearn.cross_validation import train_test_split

x_train , x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state =0)


#fitting multi linear regression

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train,y_train)

#predict the test set results

y_pred = regressor.predict(x_test)

#building modelfor backward elimination

import statsmodels.formula.api as sm

x = np.append(arr =np.ones((50,1)).astype(int),values =x,axis =1)

x_opt = x[:,[0,1,2,3,4,5]]

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()

regressor_OLS.summary()

x_opt = x[:,[0,1,3,4,5]]

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()

regressor_OLS.summary()

x_opt = x[:,[0,3,4,5]]

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()

regressor_OLS.summary()

x_opt = x[:,[0,3,5]]

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()

regressor_OLS.summary()

x_opt = x[:,[0,3]]

regressor_OLS = sm.OLS(endog =y, exog = x_opt).fit()

regressor_OLS.summary()

