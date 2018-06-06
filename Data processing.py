# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:44:33 2018

@author: sai ram
"""

import numpy as np #mathematical tools #mathematics
import matplotlib.pyplot as plt  #plottings of the data
import pandas as pd #download and manage datasets
import os

#Importing thr dataset

dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values   #all the lines, all the columns except the last one , take all the values

Y = dataset.iloc[:,3].values  #only the last column , index = 3

from sklearn.preprocessing import Imputer  #take care of missing data

imputer = Imputer(missing_values = 'NaN',strategy = "mean" ,axis = 0 )  #axis =0 gets mean of the columns


imputer = imputer.fit(X[:,1:3])  #index upper bound is excluded that's why 3

X[:,1:3] = imputer.transform(X[:,1:3])

#ENCODING CATEGORICAL variable

from sklearn.preprocessing import  LabelEncoder ,OneHotEncoder

labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])

X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()

Y = labelencoder_X.fit_transform(Y)

#SPLITTING THE DATA into test and training set

from sklearn.cross_validation import train_test_split

X_train , X_test,y_train, y_test = train_test_split(X,Y,test_size = 0.2,random_state =0)

from sklearn.preprocessing import StandardScaler

sc_x =  StandardScaler()

X_train = sc_x.fit_transform(X_train)

X_test = sc_x.fit_transform(X_test)




