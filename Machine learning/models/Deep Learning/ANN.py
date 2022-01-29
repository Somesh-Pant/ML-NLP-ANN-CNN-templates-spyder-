# Artificial Neural Network 

# Part 1 - Data Preprocessing 

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

# Feature Scaling
sc_X = preprocessing.StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - ANN

# Importing Keras
import keras
from tensorflow.python.keras.models import Sequential # To initialize the neural network
from tensorflow.python.keras.layers import Dense # To generate the layers of the neural network 

# Initialising the neural network 
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#adding the second hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu')) 
# adding the output layer 
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
# above 'relu' refers to rectifier function which is usedin the input and hidden layers whereas sigmoid is used for the output layer

# Compiling the ANN
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the Ann to the training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 1000)



#Predicting the Test Results using the classifier
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
##Evaluating the efficiency and the accuracy of the classifier##

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  #Here the first parameter specifies the real values for themodel needs to work i.e. the test set
