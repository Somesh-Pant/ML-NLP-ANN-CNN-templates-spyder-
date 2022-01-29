#Multiple linear regression

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values #Matrix of feature / Independent variable
#Here -1 removes the last coloumn of the original dataset as it is a dependent variable
y = dataset.iloc[:,4].values #Matrix of Dependent variable

#Encoding Categorical data
from sklearn import preprocessing 
labelencoder_X = preprocessing.LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
#Dummy encoding 
onehotencoder = preprocessing.OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy variable trap
X = X[:, 1:] #The dataset now will not consider the first dummy variable 
#The above line is not required , this is to highligh the dummy trap problem in regrsseion

#Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
'''from sklearn import preprocessing
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''

#Fitting Multiple Regression in the data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test results
y_pred = regressor.predict(X_test)

#Building anoptimal model using backward elemination
import statsmodels.api as sm
#The below line adds a coloumn of 1s at the bigenning of the dataset in order to relate the constant bo with independent variable xo = 1 
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:,[0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()