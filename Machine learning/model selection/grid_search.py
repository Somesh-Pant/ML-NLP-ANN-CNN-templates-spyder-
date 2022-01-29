# Grid Search

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
X_test = sc_X.transform(X_test)

#Fitting Kernel SVM Classifier to the training set
from sklearn.svm import SVC
Classifier = SVC(kernel = 'rbf', random_state = 0)
Classifier.fit(X_train, y_train)

#Predicting the Test Results using the classifier
y_pred = Classifier.predict(X_test)

# Applying the K-fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Classifier, X = X_train, y = y_train, cv = 10 )
accuracies.mean()       # Average or mean of the iterations
accuracies.std()        # Standard deviation within the iterations

# Applying grid search to find the optimal model and optimal parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [1, 10, 100, 1000], 'kernel' : ['linear']}, 
             {'C' : [1, 10, 100, 1000], 'kernel' : ['rbf'], 'gamma' : [0.5, 0.1, 0.01, 0.001, 0.0001]}
             ]          # Setup of the dictionary that is a set of values to apply on the dataset and determine the best combination of values
grid_search = GridSearchCV(estimator = Classifier, 
                           param_grid = parameters, scoring = 'accuracy', 
                           cv = 10, n_jobs = -1)
grid_search.fit(X_train , y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
 
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
plt.title("Kernel SVM (Training set)")
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
plt.title("Kernal SVM (Test set)")
plt.xlabel("Age(average)")
plt.ylabel("Salary(average)")
plt.legend()
plt.show()
