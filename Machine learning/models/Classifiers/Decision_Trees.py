#Decision Tree Classifier

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn import preprocessing
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)## feature scaling is not required for decision tree classifier 
##feature scaling is used here as the plots are being plot in a high resolution of 0.01

#Fitting Decision Tree Classifier to the training set
from sklearn.tree import DecisionTreeClassifier
Classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
Classifier.fit(X_train, y_train)

#Predicting the Test Results using the classifier
y_pred = Classifier.predict(X_test)

##Evaluating the efficiency and the accuracy of the classifier##

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  #Here the first parameter specifies the real values for themodel needs to work i.e. the test set

#Visualising the training results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# this line splits the graph in to pixels with a resolution of 0.01
plt.contourf(X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# This line applies the classifier to each pixel and colors it accordingly 
plt.xlim(X1.min(), X1.max())    #axis definitions
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
#plot formation
plt.title("Decision Tree (Training set)")
plt.xlabel("Age(average)")
plt.ylabel("Salary(average)")
plt.legend()
plt.show()

#Visualising the test results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# this line splits the graph in to pixels with a resolution of 0.01
plt.contourf(X1, X2, Classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# This line applies the classifier to each pixel and colors it accordingly 
plt.xlim(X1.min(), X1.max())    #axis definitions
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
#plot formation
plt.title("Decision Tree (Test set)")
plt.xlabel("Age(average)")
plt.ylabel("Salary(average)")
plt.legend()
plt.show()
