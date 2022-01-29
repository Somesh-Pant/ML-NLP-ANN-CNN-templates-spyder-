import pandas as pd

npr = pd.read_csv('npr.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df= 0.9, min_df = 10, stop_words = 'english')
dtm = tfidf.fit_transform(npr['Article'])

#NMF
from sklearn.decomposition import NMF
nmf_model = NMF(n_components = 10, random_state =42)
nmf_model.fit(dtm)

for index,topic in enumerate(nmf_model.components_):
    print(f"The top 15 words for topic # {index}")
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
topic_results = nmf_model.transform(dtm)

npr['Topic'] = topic_results.argmax(axis = 1)
