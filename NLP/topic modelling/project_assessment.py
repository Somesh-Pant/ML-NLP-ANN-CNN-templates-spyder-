import pandas as pd

ques = pd.read_csv('quora_questions.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df = 0.9, min_df = 5, stop_words = 'english')
dtm  = tfidf.fit_transform(ques['Question'])

from sklearn.decomposition import NMF
nmf_model = NMF(n_components = 20, random_state = 42)
nmf_model.fit(dtm)

for index,topic in enumerate(nmf_model.components_):
    print(f"THE TOP 15 WORDS FOR THE TOPIC #{index}")
    print([tfidf.get_feature_names()[index] for index in topic.argsort()[-15:]])
    print('\n')
    print('\n')
    
topic_results = nmf_model.transform(dtm)
ques['topic'] = topic_results.argmax(axis = 1)