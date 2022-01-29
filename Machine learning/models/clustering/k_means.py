#K means clustering

##Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values  #Taking only the coloumns of interest here

#Using the elbow method to predict the right numnber of clusters 
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
    #the init value that is k-means++ helps to avoid the random initialization trap
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia_ is a method used to compute the wcss for a particlar k

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters K')
plt.ylabel('WCSS')
plt.show()

# applying the k-means clustering to the dataset with right number of clusters 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = kmeans.fit_predict(X) #prediction by our clustering model

# Visualising the k-means clustering results
plt.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s = 100 , c = 'red', label = 'Cluster 1')
plt.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s = 100 , c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s = 100 , c = 'green', label = 'Cluster 3')
plt.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s = 100 , c = 'yellow', label = 'Cluster 4')
plt.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s = 100 , c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'orange', label = 'centroids' )
plt.title('k-means clusters')
plt.xlabel('Anual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.show()
