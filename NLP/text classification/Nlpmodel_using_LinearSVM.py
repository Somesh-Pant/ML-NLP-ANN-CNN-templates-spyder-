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
X = sms['message']
y = sms['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

'''
# Count Vectorizing the text in the document (I.e. Creating a vocabulary or just counting the number of different words in the text)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
#Fitting the count veoctrizer on the document
X_train_counts = cv.fit_transform(X_train)

#Applying the IF compensation
from sklearn.feature_extraction.text import TfidfTransformer
tf = TfidfTransformer()
# Fitting it to the count vectorizer
X_train_countsif = tf.fit_transform(X_train_counts) 
'''
# Direct  way to do the above task
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X_train_tf = tf.fit_transform(X_train)

# creating a classifier
from sklearn.svm import LinearSVC
clf = LinearSVC()
clf.fit(X_train_tf, y_train)

# A single step to perform the above tasks in one go is the use of the pipeline
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)

#analysis
from sklearn.metrics import confusion_matrix, classification_report , accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
