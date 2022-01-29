import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

data = pd.read_csv('moviereviews2.tsv', sep = '\t')

data.dropna(inplace = True) # To remove all the nan(not a number values) values

# To remove empty strings
blanks = []
for i, lb, rv in data.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)
data.drop(blanks, inplace = True)

X = data['review']
y = data['label']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])

text_clf.fit(X_train, y_train)

y_pred = text_clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
