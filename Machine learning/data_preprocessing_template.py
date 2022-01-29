#Data preprocessing

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the Dataset 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values #Matrix of feature / Independent variable
#Here -1 removes the last coloumn of the original dataset as it is a dependent variable
y = dataset.iloc[:,3].values #Matrix of Dependent variable

#Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
'''from sklearn import preprocessing
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)'''
