# XG boost

# Importing the libraries 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

# importing the dataset 
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values 

#Encoding Categorical data
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer 
labelencoder_X_2 = preprocessing.LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#Dummy encoding in case of multiple variable
ct = ColumnTransformer([("Country", preprocessing.OneHotEncoder(), [1])], remainder = 'passthrough')
X = ct.fit_transform(X)
#Noticed how the first LabelEncoder was removed, you do not need to apply both the label encoded and the one hot encoder on the column anymore
X = X[:, 1:]#removing a dummy variable to avoid dummy variable trap


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Applying XG Boosting to the dataset
import xgboost
Classifier = xgboost.XGBClassifier()
Classifier.fit(X_train, y_train)

#Predicting the Test Results using the classifier
y_pred = Classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 

# Applying the K-fold Cross Validation 
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Classifier, X = X_train, y = y_train, cv = 10 )
accuracies.mean()
accuracies.std()