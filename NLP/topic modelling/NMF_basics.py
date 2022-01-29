#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Non-Negative Matric Factorization
# 
# Let's repeat thet opic modeling task from the previous lecture, but this time, we will use NMF instead of LDA.

# ## Data
# 
# We will be using articles scraped from NPR (National Public Radio), obtained from their website [www.npr.org](http://www.npr.org)

# In[1]:


import pandas as pd


# In[2]:


npr = pd.read_csv('npr.csv')


# In[3]:


npr.head()


# Notice how we don't have the topic of the articles! Let's use LDA to attempt to figure out clusters of the articles.

# ## Preprocessing

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer


# **`max_df`**` : float in range [0.0, 1.0] or int, default=1.0`<br>
# When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# 
# **`min_df`**` : float in range [0.0, 1.0] or int, default=1`<br>
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

# In[5]:


tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[6]:


dtm = tfidf.fit_transform(npr['Article'])


# In[7]:


dtm


# ## NMF

# In[8]:


from sklearn.decomposition import NMF


# In[11]:


nmf_model = NMF(n_components=7,random_state=42)


# In[12]:


# This can take awhile, we're dealing with a large amount of documents!
nmf_model.fit(dtm)


# ## Displaying Topics

# In[13]:


len(tfidf.get_feature_names())


# In[14]:


import random


# In[15]:


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(tfidf.get_feature_names()[random_word_id])


# In[16]:


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(tfidf.get_feature_names()[random_word_id])


# In[19]:


len(nmf_model.components_)


# In[21]:


nmf_model.components_


# In[22]:


len(nmf_model.components_[0])


# In[23]:


single_topic = nmf_model.components_[0]


# In[24]:


# Returns the indices that would sort this array.
single_topic.argsort()


# In[25]:


# Word least representative of this topic
single_topic[18302]


# In[26]:


# Word most representative of this topic
single_topic[42993]


# In[49]:


# Top 10 words for this topic:
single_topic.argsort()[-10:]


# In[27]:


top_word_indices = single_topic.argsort()[-10:]


# In[28]:


for index in top_word_indices:
    print(tfidf.get_feature_names()[index])


# These look like business articles perhaps... Let's confirm by using .transform() on our vectorized articles to attach a label number. But first, let's view all the 10 topics found.

# In[30]:


for index,topic in enumerate(nmf_model.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# ### Attaching Discovered Topic Labels to Original Articles

# In[31]:


dtm


# In[32]:


dtm.shape


# In[33]:


len(npr)


# In[34]:


topic_results = nmf_model.transform(dtm)


# In[35]:


topic_results.shape


# In[36]:


topic_results[0]


# In[37]:


topic_results[0].round(2)


# In[38]:


topic_results[0].argmax()


# This means that our model thinks that the first article belongs to topic #1.

# ### Combining with Original Data

# In[39]:


npr.head()


# In[40]:


topic_results.argmax(axis=1)


# In[41]:


npr['Topic'] = topic_results.argmax(axis=1)


# In[42]:


npr.head(10)


# ## Great work!

