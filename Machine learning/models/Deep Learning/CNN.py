# Convolution Neural Network

# Import Theano, Tensorflow and Keras

# Part 1 - Building the CNN

# Importingthe libraries
import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Convolution2D
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution 
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer to increase the test set accuracy 
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(128, activation = 'relu'))  # Hidden layer 
classifier.add(Dense(1, activation = 'sigmoid')) # Output layer 

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
#Image augmentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# Test images rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Training Set Generator
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# Test Set Generator
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

#Fitting the model to the image
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
