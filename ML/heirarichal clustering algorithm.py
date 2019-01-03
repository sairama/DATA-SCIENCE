# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 10:11:27 2018

@author: sventrapragada
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Mall_Customers.csv')

X= dataset.iloc[:,[3,4]].values

#USING THE DENTOGRAM FIND THE OPTIMAL NUMBER OF CLUSTERS
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method = 'ward')) #ward minimize the variance within each cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distance')
plt.show()

#fitting herirarichal clustering to the mall dataset

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters  =5,affinity ='euclidean' ,linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualising the herirarichal clusters

plt.scatter(X[y_hc ==0,0],X[y_hc==0,1],s=100,c='red',label = 'Careful')
plt.scatter(X[y_hc ==1,0],X[y_hc ==1,1],s=100,c='blue',label = 'standard')
plt.scatter(X[y_hc ==2,0],X[y_hc ==2,1],s=100,c='green',label = 'potential target')
plt.scatter(X[y_hc ==3,0],X[y_hc ==3,1],s=100,c='cyan',label = 'Careless')
plt.scatter(X[y_hc ==4,0],X[y_hc ==4,1],s=100,c='magenta',label = 'sensible')
plt.title('clusters of clients')
plt.xlabel('annaual income')
plt.ylabel('spending_score[1-100]')
plt.legend()
plt.show()
