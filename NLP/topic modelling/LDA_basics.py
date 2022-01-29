#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Latent Dirichlet Allocation

# ## Data
# 
# We will be using articles from NPR (National Public Radio), obtained from their website [www.npr.org](http://www.npr.org)

# In[1]:


import pandas as pd


# In[2]:


npr = pd.read_csv('npr.csv')


# In[3]:


npr.head()


# Notice how we don't have the topic of the articles! Let's use LDA to attempt to figure out clusters of the articles.

# ## Preprocessing

# In[4]:


from sklearn.feature_extraction.text import CountVectorizer


# **`max_df`**` : float in range [0.0, 1.0] or int, default=1.0`<br>
# When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words). If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.
# 
# **`min_df`**` : float in range [0.0, 1.0] or int, default=1`<br>
# When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold. This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

# In[5]:


cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')


# In[6]:


dtm = cv.fit_transform(npr['Article'])


# In[7]:


dtm


# ## LDA

# In[8]:


from sklearn.decomposition import LatentDirichletAllocation


# In[39]:


LDA = LatentDirichletAllocation(n_components=7,random_state=42)


# In[40]:


# This can take awhile, we're dealing with a large amount of documents!
LDA.fit(dtm)


# ## Showing Stored Words

# In[41]:


len(cv.get_feature_names())


# In[42]:


import random


# In[43]:


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])


# In[44]:


for i in range(10):
    random_word_id = random.randint(0,54776)
    print(cv.get_feature_names()[random_word_id])


# ### Showing Top Words Per Topic

# In[46]:


len(LDA.components_)


# In[47]:


LDA.components_


# In[48]:


len(LDA.components_[0])


# In[49]:


single_topic = LDA.components_[0]


# In[50]:


# Returns the indices that would sort this array.
single_topic.argsort()


# In[51]:


# Word least representative of this topic
single_topic[18302]


# In[52]:


# Word most representative of this topic
single_topic[42993]


# In[53]:


# Top 10 words for this topic:
single_topic.argsort()[-10:]


# In[54]:


top_word_indices = single_topic.argsort()[-10:]


# In[55]:


for index in top_word_indices:
    print(cv.get_feature_names()[index])


# These look like business articles perhaps... Let's confirm by using .transform() on our vectorized articles to attach a label number. But first, let's view all the 10 topics found.

# In[56]:


for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names()[i] for i in topic.argsort()[-15:]])
    print('\n')


# ### Attaching Discovered Topic Labels to Original Articles

# In[57]:


dtm


# In[58]:


dtm.shape


# In[59]:


len(npr)


# In[60]:


topic_results = LDA.transform(dtm)


# In[61]:


topic_results.shape


# In[62]:


topic_results[0]


# In[63]:


topic_results[0].round(2)


# In[64]:


topic_results[0].argmax()


# This means that our model thinks that the first article belongs to topic #1.

# ### Combining with Original Data

# In[65]:


npr.head()


# In[66]:


topic_results.argmax(axis=1)


# In[67]:


npr['Topic'] = topic_results.argmax(axis=1)


# In[68]:


npr.head(10)


# ## Great work!

