# Hiearchical Clustering

##Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  #Taking only the coloumns of interest here

# Use of dendrograms ti find the optimal number of Clusters
import scipy.cluster.hierarchy as sch
Dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#Here linkage is the main algorithm and the ward method is used to execute it in order to minimize the Within Cluster Variance WCV like WCCS
# WCV is the variance within each cluster
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show() 

# Fitting the hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
# affinity defines the method to claculate the dissimilarities and the linkage is ward again to minimize ward
y_hc = hc.fit_predict(X) #PRediction via the model
# The cluster index starts form 0 

#Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100 , c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100 , c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100 , c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100 , c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100 , c = 'magenta', label = 'Cluster 5')
plt.title('Agglomerative Hierarchical clusters')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.show()
