# Polynomial Regression

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

#Fitting Linear Regression to dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Fitting Polynomial Regression to the model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising Linear Regression Results
plt.scatter(X, y, color = 'Red')
plt.plot(X, lin_reg.predict(X), color = 'Blue')
plt.title("Linear regression visualisation")
plt.xlabel("Level")
plt.ylabel("Salary")

#Polynomial Regression Results Visualisation
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
#Theabove two lines are to make the curve more smooth and continous
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'Blue')
plt.title("Polynomial regression visualisation")
plt.xlabel("Level")
plt.ylabel("Salary")

#Prediction of linear regression 
lin_reg.predict([[6.5]])
#Prediction of plynomial regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))