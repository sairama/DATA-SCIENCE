# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 20:48:45 2018

@author: sai ram
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import keras
# Importing the dataset
os.chdir('G:\\Machine_Learning_AZ_Template_Folder\\Machine Learning A-Z Template Folder\\Part 8 - Deep Learning\\Section 39 - Artificial Neural Networks (ANN)')
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X =X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#import keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

#initialize ann
classifier  = Sequential()

#adding the input layer an first hidden layer
classifier.add(Dense(output_dim =6, init ='uniform',activation ='relu',input_dim=11))
#adding the second hidden layer
classifier.add(Dense(output_dim =6, init ='uniform',activation ='relu'))
#add the output layer
classifier.add(Dense(output_dim =1, init ='uniform',activation ='sigmoid'))

#compiling the ann
classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics = ['accuracy'])

#fitting the classifier to the training set

classifier.fit(X_train, y_train, batch_size =10, nb_epoch = 100)

#create the classifer


#predict the test set reults
y_pred = classifier.predict(X_test)
y_pred =(y_pred>0.5)
#computing confusion matrix

from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(y_test,y_pred)
