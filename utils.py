
import re
from nltk.tokenize import word_tokenize
import random
import nltk
from collections import Counter, defaultdict
import pickle

import json


with open("cleaned_data.txt", "rb") as fp:   # Unpickling
    cleaned_data = pickle.load(fp)



word_count={}
for word in cleaned_data:
    if word not in word_count:
        word_count[word]=0
    word_count[word]+=1    
  










def extra_space(text):
    
    new_text= re.sub("\s+"," ",text)
    return new_text
    

def sp_charac(text):
    
    new_text=re.sub("[^0-9A-Za-z ]", "" , text)
    return new_text
    

def tokenize_text(text):
    
    new_text=word_tokenize(text)
    return new_text
    
  

    
  
import numpy as np
np.save('unigram_dict', word_count) 
    

ngram_dict1=defaultdict(lambda: defaultdict(lambda: 0))



def gen_uni_keys(cleaned_data):
    
    for i in range(len(cleaned_data) - 1):
        
          yield [ cleaned_data[i], cleaned_data[i + 1] ]


uni_pairs=gen_uni_keys(cleaned_data)
for pair in uni_pairs:
    ngram_dict1[pair[0]][pair[1]]+=1    
  
with open('uni_dict.json', 'w') as fp:
    json.dump(ngram_dict1, fp)


with open('uni_dict.json', 'r') as fp:
    uni_dict=json.loads( fp.read())


def prob(word_count):
    for words in word_count:
        total_count = float(sum(word_count[words].values()))
        for nw in word_count[words]:
            word_count[words][nw] = (word_count[words][nw]+1)/(total_count+vocab)
    



    



def unipred(word):
    
   if word not in uni_dict.keys(): #if that word does not exist in our dictionary then we predict some random word
        
        preds=Counter(ngram_dict1[random.choice(list(counter.keys()))]).most_common()[:3]
        print(preds)
   else:
        preds=Counter(uni_dict[word]).most_common()[:3]
        print(preds)
        
     




