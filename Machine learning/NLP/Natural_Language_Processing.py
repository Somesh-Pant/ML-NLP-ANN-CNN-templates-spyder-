# NLP 

# importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the dataset 
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#the delimiter parameter is used to read the tsv (tab spaced values) and the quoting parameter is used to ignore double quotes

# Cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords # downloading and importing the listof undesirable words suchas this,the,that etc.
from nltk.stem.porter import PorterStemmer #imprting the class for stemming
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ' ,dataset['Review'][i]) #This tool removes all the characters that are not mentioned in the parameters from a review
    review = review.lower() # to convert the letters to lower case only
    review = review.split() # splitting the review into words
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # comparison with stopwords and identifying meaningful text
    review = ' '.join(review) # converting the cleaned list back to the string to get a clear text
    corpus.append(review)
    
# Creating the Bag fo Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) # for tokenization and the max-features parameter is used for reduceing the dimension by considering only frequently used words (1500 most used words in this case) 
X = cv.fit_transform(corpus).toarray() # for creating the sparce matrix
y = dataset.iloc[:, 1].values  # Dependent variable

# Naive Bayes Classification 
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting Naive Bayes Classifier to the training set
from sklearn.naive_bayes import GaussianNB
Classifier = GaussianNB()
Classifier.fit(X_train, y_train)

#Predicting the Test Results using the classifier
y_pred = Classifier.predict(X_test)

##Evaluating the efficiency and the accuracy of the classifier##

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  #Here the first parameter specifies the real values for themodel needs to work i.e. the test set
