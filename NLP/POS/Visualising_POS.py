#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Visualizing Parts of Speech
# spaCy offers an outstanding visualizer called **displaCy**:

# In[1]:


# Perform standard imports
import spacy
nlp = spacy.load('en_core_web_sm')

# Import the displaCy library
from spacy import displacy


# In[2]:


# Create a simple Doc object
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")


# In[3]:


# Render the dependency parse immediately inside Jupyter:
displacy.render(doc, style='dep', jupyter=True, options={'distance': 110})


# The dependency parse shows the coarse POS tag for each token, as well as the **dependency tag** if given:

# In[13]:


for token in doc:
    print(f'{token.text:{10}} {token.pos_:{7}} {token.dep_:{7}} {spacy.explain(token.dep_)}')


# ___
# # Creating Visualizations Outside of Jupyter
# If you're using another Python IDE or writing a script, you can choose to have spaCy serve up HTML separately.
# 
# Instead of `displacy.render()`, use `displacy.serve()`:

# In[15]:


displacy.serve(doc, style='dep', options={'distance': 110})


# <font color=blue>**After running the cell above, click the link below to view the dependency parse**:</font>
# 
# http://127.0.0.1:5000
# <br><br>
# <font color=red>**To shut down the server and return to jupyter**, interrupt the kernel either through the **Kernel** menu above, by hitting the black square on the toolbar, or by typing the keyboard shortcut `Esc`, `I`, `I`</font>

# <font color=green>**NOTE**: We'll use this method moving forward because, at this time, several of the customizations we want to show don't work well in Jupyter.</font>

# ___
# ## Handling Large Text
# `displacy.serve()` accepts a single Doc or list of Doc objects. Since large texts are difficult to view in one line, you may want to pass a list of spans instead. Each span will appear on its own line:

# In[16]:


doc2 = nlp(u"This is a sentence. This is another, possibly longer sentence.")

# Create spans from Doc.sents:
spans = list(doc2.sents)

displacy.serve(spans, style='dep', options={'distance': 110})


# **Click this link to view the dependency**: http://127.0.0.1:5000
# <br>Interrupt the kernel to return to jupyter.

# ___
# ## Customizing the Appearance
# Besides setting the distance between tokens, you can pass other arguments to the `options` parameter:
# 
# <table>
# <tr><th>NAME</th><th>TYPE</th><th>DESCRIPTION</th><th>DEFAULT</th></tr>
# <tr><td>`compact`</td><td>bool</td><td>"Compact mode" with square arrows that takes up less space.</td><td>`False`</td></tr>
# <tr><td>`color`</td><td>unicode</td><td>Text color (HEX, RGB or color names).</td><td>`#000000`</td></tr>
# <tr><td>`bg`</td><td>unicode</td><td>Background color (HEX, RGB or color names).</td><td>`#ffffff`</td></tr>
# <tr><td>`font`</td><td>unicode</td><td>Font name or font family for all text.</td><td>`Arial`</td></tr>
# </table>
# 
# For a full list of options visit https://spacy.io/api/top-level#displacy_options

# In[17]:


options = {'distance': 110, 'compact': 'True', 'color': 'yellow', 'bg': '#09a3d5', 'font': 'Times'}

displacy.serve(doc, style='dep', options=options)


# **Click this link to view the dependency**: http://127.0.0.1:5000
# <br>Interrupt the kernel to return to jupyter.

# ___
# Great! Now you should be familiar with visualizing spaCy's dependency parse. For more info on **displaCy** visit https://spacy.io/usage/visualizers
# <br>In the next section we'll look at Named Entity Recognition, followed by displaCy's NER visualizer.
# 
# ### Next Up: Named Entity Recognition

