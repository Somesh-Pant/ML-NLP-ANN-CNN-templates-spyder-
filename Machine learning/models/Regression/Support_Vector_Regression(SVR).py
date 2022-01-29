#Support Vector Regression

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Matrix of feature / Independent variable
y = dataset.iloc[:, 2].values #Matrix of Dependent variable
y = np.array(y).reshape(-1, 1) # For feature scaling to work

#Splitting the dataset into test set and training set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#No splitting required as the dataset is very small

#Feature Scaling
from sklearn import preprocessing
sc_X = preprocessing.StandardScaler()
sc_y = preprocessing.StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Fitting Support Vector Regression model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)
#Create your regressor here

#Prediction of plynomial regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))
#The above line calculate the prediction in feature scaling and takes it inverse transform to return the prediction in original form

#Support Vector Regression Results Visualisation
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
#Theabove two lines are to make the curve more smooth and continous
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'Blue')
plt.title("Support Vector regression visualisation")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()
