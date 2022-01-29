#Regularization applied on Linear Regression

#Importing Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
                                                    
#Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Using Ridge Regression
from sklearn.linear_model import Ridge
rr = Ridge(alpha = 100)#large value of alpha is taken to display the difference in the ridge and linear regeression
rr.fit(X_train, y_train)

#Using Lasso regression
from sklearn.linear_model import Lasso
lr = Lasso(alpha = 10000)#alpha here is taken larger as the number of parameters are small and no. of useless parameters are very low
#10000 is taken to clearly visualize lasso regrssion on the plot
lr.fit(X_train, y_train)

#Visualising the training results 
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.plot(X_train, rr.predict(X_train), color = 'blue')
plt.plot(X_train, lr.predict(X_train), color = 'yellow')
plt.title("salary vs experience (training set)")
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()

