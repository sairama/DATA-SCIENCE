# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 23:29:28 2018

@author: sai ram
"""
#import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t',quoting =3)

#cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i] ) #includes all the letters and spaces
    review = review.lower()
    review = review.split()
    ps = PorterStemmer() 
    review =[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

#create bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
# Feature Scaling

'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)''''

#fitting the classifier to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


#predict the test set reults
y_pred = classifier.predict(X_test)

#computing confusion matrix

from sklearn.metrics import confusion_matrix
cm =  confusion_matrix(y_test,y_pred)
print(cm)





