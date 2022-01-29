import pandas as pd 
import numpy as np

sms = pd.read_csv('smsspamcollection.tsv', sep = '\t')

# code block to visualize the dataset in the form of a histogram
import matplotlib.pyplot as plt

plt.xscale('log')
bins = 1.15**(np.arange(0,50))
plt.hist(sms[sms['label']=='ham']['length'],bins=bins,alpha=0.8)
plt.hist(sms[sms['label']=='spam']['length'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

# visualising the punctuations in spam and ham messages
plt.xscale('log')
bins = 1.5**(np.arange(0,15))
plt.hist(sms[sms['label']=='ham']['punct'],bins=bins,alpha=0.8)
plt.hist(sms[sms['label']=='spam']['punct'],bins=bins,alpha=0.8)
plt.legend(('ham','spam'))
plt.show()

from sklearn.model_selection import train_test_split 
# defining the X feature 
X = sms[['length','punct']]
y = sms['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


#Fitting Naive Bayes Classifier to the training set
from sklearn.naive_bayes import MultinomialNB
Classifier = MultinomialNB()
Classifier.fit(X_train, y_train)

#Predicting the Test Results using the classifier
y_pred = Classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test, y_pred)
#labeling the confusion matrix
sms = pd.DataFrame(cm, index=['ham','spam'], columns=['ham','spam'])

#creating a classification report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)

# getting the accuracy score
acc = accuracy_score(y_test, y_pred)