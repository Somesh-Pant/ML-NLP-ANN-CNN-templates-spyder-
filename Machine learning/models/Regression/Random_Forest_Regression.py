#Random_Forest Regression
#Regression template

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #Matrix of feature / Independent variable
y = dataset.iloc[:, 2].values #Matrix of Dependent variable

#Splitting the dataset into test set and training set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
#No splitting required as the dataset is very small

#Feature Scaling
'''from sklearn import preprocessing
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#Create your regressor here

#Prediction of random forest regression
y_pred = regressor.predict([[6.5]])

#Random Forest Regression Regression Results Visualisation
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
#Theabove two lines are to make the curve more smooth and continous
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'Blue')
plt.title("Random Forest regression visualisation")
plt.xlabel("Level")
plt.ylabel("Salary")