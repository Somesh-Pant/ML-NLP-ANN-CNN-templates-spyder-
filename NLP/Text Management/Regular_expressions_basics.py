#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <a href='http://www.pieriandata.com'> <img src='../Pierian_Data_Logo.png' /></a>
# ___

# # Regular Expressions
# 
# Regular Expressions (sometimes called regex for short) allow a user to search for strings using almost any sort of rule they can come up with. For example, finding all capital letters in a string, or finding a phone number in a document. 
# 
# Regular expressions are notorious for their seemingly strange syntax. This strange syntax is a byproduct of their flexibility. Regular expressions have to be able to filter out any string pattern you can imagine, which is why they have a complex string pattern format.
# 
# Regular expressions are handled using Python's built-in **re** library. See [the docs](https://docs.python.org/3/library/re.html) for more information.

# Let's begin by explaining how to search for basic patterns in a string!

# ## Searching for Basic Patterns
# 
# Let's imagine that we have the following string:

# In[1]:


text = "The agent's phone number is 408-555-1234. Call soon!"


# We'll start off by trying to find out if the string "phone" is inside the text string. Now we could quickly do this with:

# In[2]:


'phone' in text


# But let's show the format for regular expressions, because later on we will be searching for patterns that won't have such a simple solution.

# In[3]:


import re


# In[4]:


pattern = 'phone'


# In[5]:


re.search(pattern,text)


# In[6]:


pattern = "NOT IN TEXT"


# In[7]:


re.search(pattern,text)


# Now we've seen that re.search() will take the pattern, scan the text, and then returns a Match object. If no pattern is found, a None is returned (in Jupyter Notebook this just means that nothing is output below the cell).
# 
# Let's take a closer look at this Match object.

# In[8]:


pattern = 'phone'


# In[9]:


match = re.search(pattern,text)


# In[10]:


match


# Notice the span, there is also a start and end index information.

# In[11]:


match.span()


# In[12]:


match.start()


# In[13]:


match.end()


# But what if the pattern occurs more than once?

# In[14]:


text = "my phone is a new phone"


# In[15]:


match = re.search("phone",text)


# In[16]:


match.span()


# Notice it only matches the first instance. If we wanted a list of all matches, we can use .findall() method:

# In[17]:


matches = re.findall("phone",text)


# In[18]:


matches


# In[19]:


len(matches)


# To get actual match objects, use the iterator:

# In[20]:


for match in re.finditer("phone",text):
    print(match.span())


# If you wanted the actual text that matched, you can use the .group() method.

# In[21]:


match.group()


# # Patterns
# 
# So far we've learned how to search for a basic string. What about more complex examples? Such as trying to find a telephone number in a large string of text? Or an email address?
# 
# We could just use search method if we know the exact phone or email, but what if we don't know it? We may know the general format, and we can use that along with regular expressions to search the document for strings that match a particular pattern.
# 
# This is where the syntax may appear strange at first, but take your time with this; often it's just a matter of looking up the pattern code.
# 
# Let's begin!

# ## Identifiers for Characters in Patterns
# 
# Characters such as a digit or a single string have different codes that represent them. You can use these to build up a pattern string. Notice how these make heavy use of the backwards slash \ . Because of this when defining a pattern string for regular expression we use the format:
# 
#     r'mypattern'
#     
# placing the r in front of the string allows python to understand that the \ in the pattern string are not meant to be escape slashes.
# 
# Below you can find a table of all the possible identifiers:

# <table ><tr><th>Character</th><th>Description</th><th>Example Pattern Code</th><th >Exammple Match</th></tr>
# 
# <tr ><td><span >\d</span></td><td>A digit</td><td>file_\d\d</td><td>file_25</td></tr>
# 
# <tr ><td><span >\w</span></td><td>Alphanumeric</td><td>\w-\w\w\w</td><td>A-b_1</td></tr>
# 
# 
# 
# <tr ><td><span >\s</span></td><td>White space</td><td>a\sb\sc</td><td>a b c</td></tr>
# 
# 
# 
# <tr ><td><span >\D</span></td><td>A non digit</td><td>\D\D\D</td><td>ABC</td></tr>
# 
# <tr ><td><span >\W</span></td><td>Non-alphanumeric</td><td>\W\W\W\W\W</td><td>*-+=)</td></tr>
# 
# <tr ><td><span >\S</span></td><td>Non-whitespace</td><td>\S\S\S\S</td><td>Yoyo</td></tr></table>

# For example:

# In[22]:


text = "My telephone number is 408-555-1234"


# In[23]:


phone = re.search(r'\d\d\d-\d\d\d-\d\d\d\d',text)


# In[24]:


phone.group()


# Notice the repetition of \d. That is a bit of an annoyance, especially if we are looking for very long strings of numbers. Let's explore the possible quantifiers.
# 
# ## Quantifiers
# 
# Now that we know the special character designations, we can use them along with quantifiers to define how many we expect.

# <table ><tr><th>Character</th><th>Description</th><th>Example Pattern Code</th><th >Exammple Match</th></tr>
# 
# <tr ><td><span >+</span></td><td>Occurs one or more times</td><td>	Version \w-\w+</td><td>Version A-b1_1</td></tr>
# 
# <tr ><td><span >{3}</span></td><td>Occurs exactly 3 times</td><td>\D{3}</td><td>abc</td></tr>
# 
# 
# 
# <tr ><td><span >{2,4}</span></td><td>Occurs 2 to 4 times</td><td>\d{2,4}</td><td>123</td></tr>
# 
# 
# 
# <tr ><td><span >{3,}</span></td><td>Occurs 3 or more</td><td>\w{3,}</td><td>anycharacters</td></tr>
# 
# <tr ><td><span >\*</span></td><td>Occurs zero or more times</td><td>A\*B\*C*</td><td>AAACC</td></tr>
# 
# <tr ><td><span >?</span></td><td>Once or none</td><td>plurals?</td><td>plural</td></tr></table>

# Let's rewrite our pattern using these quantifiers:

# In[25]:


re.search(r'\d{3}-\d{3}-\d{4}',text)


# ## Groups
# 
# What if we wanted to do two tasks, find phone numbers, but also be able to quickly extract their area code (the first three digits). We can use groups for any general task that involves grouping together regular expressions (so that we can later break them down). 
# 
# Using the phone number example, we can separate groups of regular expressions using parentheses:

# In[26]:


phone_pattern = re.compile(r'(\d{3})-(\d{3})-(\d{4})')


# In[27]:


results = re.search(phone_pattern,text)


# In[28]:


# The entire result
results.group()


# In[29]:


# Can then also call by group position.
# remember groups were separated by parentheses ()
# Something to note is that group ordering starts at 1. Passing in 0 returns everything
results.group(1)


# In[30]:


results.group(2)


# In[31]:


results.group(3)


# In[32]:


# We only had three groups of parentheses
results.group(4)


# ## Additional Regex Syntax
# 
# ### Or operator |
# 
# Use the pipe operator to have an **or** statment. For example

# In[33]:


re.search(r"man|woman","This man was here.")


# In[34]:


re.search(r"man|woman","This woman was here.")


# ### The Wildcard Character
# 
# Use a "wildcard" as a placement that will match any character placed there. You can use a simple period **.** for this. For example:

# In[35]:


re.findall(r".at","The cat in the hat sat here.")


# In[36]:


re.findall(r".at","The bat went splat")


# Notice how we only matched the first 3 letters, that is because we need a **.** for each wildcard letter. Or use the quantifiers described above to set its own rules.

# In[37]:


re.findall(r"...at","The bat went splat")


# However this still leads the problem to grabbing more beforehand. Really we only want words that end with "at".

# In[38]:


# One or more non-whitespace that ends with 'at'
re.findall(r'\S+at',"The bat went splat")


# ### Starts With and Ends With
# 
# We can use the **^** to signal starts with, and the **$** to signal ends with:

# In[39]:


# Ends with a number
re.findall(r'\d$','This ends with a number 2')


# In[40]:


# Starts with a number
re.findall(r'^\d','1 is the loneliest number.')


# Note that this is for the entire string, not individual words!

# ### Exclusion
# 
# To exclude characters, we can use the **^** symbol in conjunction with a set of brackets **[]**. Anything inside the brackets is excluded. For example:

# In[41]:


phrase = "there are 3 numbers 34 inside 5 this sentence."


# In[42]:


re.findall(r'[^\d]',phrase)


# To get the words back together, use a + sign 

# In[43]:


re.findall(r'[^\d]+',phrase)


# We can use this to remove punctuation from a sentence.

# In[44]:


test_phrase = 'This is a string! But it has punctuation. How can we remove it?'


# In[45]:


re.findall('[^!.? ]+',test_phrase)


# In[46]:


clean = ' '.join(re.findall('[^!.? ]+',test_phrase))


# In[47]:


clean


# ## Brackets for Grouping
# 
# As we showed above we can use brackets to group together options, for example if we wanted to find hyphenated words:

# In[48]:


text = 'Only find the hypen-words in this sentence. But you do not know how long-ish they are'


# In[49]:


re.findall(r'[\w]+-[\w]+',text)


# ## Parentheses for Multiple Options
# 
# If we have multiple options for matching, we can use parentheses to list out these options. For Example:

# In[50]:


# Find words that start with cat and end with one of these options: 'fish','nap', or 'claw'
text = 'Hello, would you like some catfish?'
texttwo = "Hello, would you like to take a catnap?"
textthree = "Hello, have you seen this caterpillar?"


# In[51]:


re.search(r'cat(fish|nap|claw)',text)


# In[52]:


re.search(r'cat(fish|nap|claw)',texttwo)


# In[53]:


# None returned
re.search(r'cat(fish|nap|claw)',textthree)


# ### Conclusion
# 
# Excellent work! For full information on all possible patterns, check out: https://docs.python.org/3/howto/regex.html

# ## Next up: Python Text Basics Assessment

