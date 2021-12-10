#!/usr/bin/env python
# coding: utf-8

# ###**Import Libraries**

# In[1]:


import pandas as pd 
import nltk
import re 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
nltk.download('punkt')
import json


# In[2]:


cd "C:/Users/onero/Pictures/images/swiftkey"


# ###**Reading Dataset**

# In[3]:


data= open('data.txt',encoding="utf8").read()


# In[4]:


print(len(data))


# ###**Cleaning Part**

# In[5]:


def extra_space(text):
    new_text= re.sub("\s+"," ",text)
    return new_text
def sp_charac(text):
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text
def tokenize_text(text):
    new_text=word_tokenize(text)
    return new_text
def tokenize_twitter(text):
    tweet = TweetTokenizer()
    new_text=tweet.tokenize(text)
    return new_text


# In[6]:


cleaned_data=extra_space(data)
print("Removed Extra Spaces")
cleaned_data=sp_charac(cleaned_data)
print("Removed Special Caracters")
cleaned_data=tokenize_text(cleaned_data)
print("Tokenized data")


# In[7]:


print(cleaned_data[:50])


# ###**Storing them in separate text files**

# In[8]:


# Storing them in separate text files
import pickle
with open("cleaned_data.json",'w') as fp:   #Pickling
    json.dump(cleaned_data, fp)


# In[9]:


import pickle
with open("cleaned_data.json", "r") as fp:   # Unpickling
    cleaned_data = json.load(fp)


# ###**Creating dictionary of unigrams with stopwords**

# In[10]:


word_count={}
for word in cleaned_data:
    if word not in word_count:
        word_count[word]=0
    word_count[word]+=1


# In[11]:


import numpy as np
np.save('unigram_dict.npy', word_count) 


# In[12]:


freq_df  = pd.DataFrame.from_dict(word_count,orient='index',columns=['Count'])
freq_df=freq_df.sort_values(by=['Count'],ascending=False)
freq_df.head()


# ###**Creating dictionary of unigrams without stopwords**

# In[21]:


pip install wordcloud


# In[13]:


from nltk.corpus import stopwords
counter={}
for i in word_count.keys():
    if i not in list(stopwords):
        counter[i]=word_count[i]
print(len(counter.keys()))


# There are a total of 561097 unique words in the corpus excluding stopwords

# ##**Visualization**

# In[59]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(50,20))
sns.barplot(freq_df.head(50).index,freq_df.head(50)['Count'])
plt.xlabel("Top 50 words")
plt.ylabel("Frequency")
plt.show()


# In[15]:


plt.figure(figsize=(100,20))
sns.barplot(freq_df.tail(50).index,freq_df.tail(50)['Count'])
plt.xlabel("Last 50 words")
plt.ylabel("Frequency")
plt.show()


# In[58]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

wordcloud = WordCloud(background_color="black").generate_from_frequencies(word_count)
plt.figure(figsize=(8, 5))
plt.axis("off")
plt.title("Unigram Wordcloud")
plt.imshow(wordcloud)
    


# In[19]:


import nltk
from collections import Counter, defaultdict


# In[20]:


ngram_dict1=defaultdict(lambda: defaultdict(lambda: 0))


# In[21]:


import pickle
with open("cleaned_data.txt", "rb") as fp:   # Unpickling
    cleaned_data = pickle.load(fp)
cleaned_corpus=cleaned_data
print(len(cleaned_corpus))


# In[22]:


#Creating a dictionary that has unigram words as the key and the word following it along with 
#its frequency of occurrence with the previous one is the value.

def gen_uni_keys(cleaned_data):
    for i in range(len(cleaned_data) - 1):
        
          yield [ cleaned_data[i], cleaned_data[i + 1] ]


# In[23]:


uni_pairs=gen_uni_keys(cleaned_data)
for pair in uni_pairs:
    ngram_dict1[pair[0]][pair[1]]+=1


# del uni_pairs

# In[24]:


#Saving it as a JSON file
import json

with open('uni_dict.json', 'w') as fp:
    json.dump(ngram_dict1, fp)


# #deleting the unnecessary files
# #del dict_temp
# del ngram_dict1

# In[25]:


# Reading the unigram dictionary
with open('uni_dict.json', 'r') as fp:
    ngram_dict1=json.load( fp)


# In[26]:


print(ngram_dict1)


# In[27]:


# Calculating vocabulary of the entire corpus
vocab=len(set(cleaned_data))
vocab


# In[28]:


#Here I ave defined the probability function where I calculate the total count of all the words succeeding the key.
#Then I perform Laplace smoothing for each of the next word by adding 1 to its count 
#and dividing by the sum of total count and vocabulary.

def prob(word_count):
    for words in word_count:
        total_count = float(sum(word_count[words].values()))
        for nw in word_count[words]:
            word_count[words][nw] = (word_count[words][nw]+1)/(total_count+vocab)
    


# In[29]:


#calculating probability
prob(ngram_dict1)


# In[30]:


#Saving the new dictionary back
with open('uni_dict.json', 'w') as fp:
    json.dump(ngram_dict1, fp)


# del ngram_dict1

# ###**Markov Model**

# In[31]:


import re 
from nltk.tokenize import word_tokenize
from collections import Counter
import time
#import orjson
import json
import random


# In[32]:


#import json
with open('uni_dict.json', 'r') as fp:
    uni_dict=json.loads( fp.read())


# In[33]:


def extra_space(text):
    new_text= re.sub("\s+"," ",text)
    return new_text
def sp_charac(text):
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text
def tokenize_text(text):
    new_text=word_tokenize(text)
    return new_text


# In[55]:



def unipred(word): #This function is used when we have only one preceding word 
    if word not in uni_dict.keys(): #if that word does not exist in our dictionary then we predict some random word
        
        preds=Counter(ngram_dict1[random.choice(list(uni_dict.keys()))])[:3]
        print(preds)
    else:
        preds=Counter(uni_dict[word]).most_common()[:3]
        print(preds)


# In[47]:


text='great'


# In[57]:


unipred(text)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




