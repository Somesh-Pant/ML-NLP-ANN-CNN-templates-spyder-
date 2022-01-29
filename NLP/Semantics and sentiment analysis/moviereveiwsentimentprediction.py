import numpy as np
import pandas as pd 

rev = pd.read_csv('moviereviews.tsv', sep ='\t')

rev.dropna(inplace=True)

blanks = []

for i,lb,rv in rev.itertuples():
    if type(rv) == str:
        if rv.isspace():
            blanks.append(i)

rev.drop(blanks,inplace=True)
        
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

rev['score'] = rev['review'].apply(lambda review: sid.polarity_scores(review))

rev['compound'] = rev['score'].apply(lambda d: d['compound'])

rev['comp_score'] = rev['compound']. apply(lambda score: 'pos' if score>=0 else 'neg')

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(rev['label'], rev['comp_score'])