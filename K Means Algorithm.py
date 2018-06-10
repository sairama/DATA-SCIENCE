# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 13:11:05 2018

@author: sai ram
"""
#importing the libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data set
dataset = pd.read_csv('Mall_Customers.csv')
X= dataset.iloc[:,[3,4]].values

#using the elbow method for optimal no of clusters
from sklearn.cluster import KMeans

wcss =[]
for i in range(1,11):
    kmeans = KMeans(n_clusters =i,init ='k-means++',max_iter = 300,n_init =10,random_state =0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('No of clusters')
plt.ylabel('WCSS')
plt.show()

#applying kmeans to the mall dataset
kmeans = KMeans(n_clusters = 5,init ='k-means++',max_iter = 300,n_init =10,random_state =0)
y_kmeans = kmeans.fit_predict(X)

#visualising the clusters

plt.scatter(X[y_kmeans ==0,0],X[y_kmeans ==0,1],s=100,c='red',label = 'Careful')
plt.scatter(X[y_kmeans ==1,0],X[y_kmeans ==1,1],s=100,c='blue',label = 'standard')
plt.scatter(X[y_kmeans ==2,0],X[y_kmeans ==2,1],s=100,c='green',label = 'potential target')
plt.scatter(X[y_kmeans ==3,0],X[y_kmeans ==3,1],s=100,c='cyan',label = 'Careless')
plt.scatter(X[y_kmeans ==4,0],X[y_kmeans ==4,1],s=100,c='magenta',label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1] ,s=300,c='yellow',label = 'Centroids')
plt.title('clusters of clients')
plt.xlabel('annaual income')
plt.ylabel('spending_score[1-100]')
plt.legend()
plt.show()

