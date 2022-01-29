import pandas as pd

npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_df = 0.9, min_df = 10, stop_words = 'english')

dtm = vect.fit_transform(npr['Article'])


#Applying LDA
from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components = 10, random_state = 42)
LDA.fit(dtm)

#  Grab the vocabulary of words
import random 
random_word_id = random.randint(0,22847)
vect.get_feature _names()[random_word_id]

# Grab the topics
random_topic_id = random.randint(0, 10)
LDA.components_[random_topic_id]

#Grab the highest probability words per topic
single_topic = LDA.components_[0]
single_topic.argsort() 

#ARGSORT -----> INDEX POSTIONS sorted from least ---> greates
#Top 10 values i.e. last 10 values of argsort

top_ten_word = single_topic.argsort()[-10:]

for index in top_ten_word:
    print(vect.get_feature_names()[index])
    
for i,topic in enumerate(LDA.components_):
    print(f"THE TOP 15 WORDS FOR THE TOPIC #{i}")
    print([vect.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')
# connecting the topics to the original dataframe
topic_results = LDA.transform(dtm)
npr['Topic'] = topic_results.argmax(axis =1)