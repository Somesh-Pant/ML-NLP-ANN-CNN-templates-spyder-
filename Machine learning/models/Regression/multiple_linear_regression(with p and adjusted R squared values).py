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
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
