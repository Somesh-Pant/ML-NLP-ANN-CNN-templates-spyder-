import numpy as np 
from sklearn.datasets import load_iris

#loading built in dataset iris
iris = load_iris()

X = iris.data

y = iris.target # grabbing the label

#Onehotencoding
from keras.utils import to_categorical
y = to_categorical(y)

# Train Test split 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

# Scaling 
from  sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
scale.fit(X_train)
scaled_X_train = scale.transform(X_train)
scaled_X_test = scale.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(8, input_dim = 4, activation = 'relu'))
model.add(Dense(8, input_dim =4, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.summary()

model.fit(scaled_X_train, y_train, epochs = 150, verbose = 2) 

y_pred = model.predict_classes(scaled_X_test)
conv_y =y_test.argmax(axis=1)
from sklearn.metrics import accuracy_score
accuracy_score(conv_y, y_pred)

model.save('myfirstmodel.h5')
from keras.models import load_model
new_model = load_model('myfirstmodel.h5')
