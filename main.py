import streamlit as st 
import pandas as pd
import numpy as np
import nltk 
import pickle
import time
import re
from nltk.tokenize import word_tokenize
from utils  import *
import json
import streamlit as st


st.title('word prediction')
st.balloons()

text= st.text_input("Enter a sentence")



 #This function is used when we have only one preceding word 

if text not in uni_dict.keys(): #if that word does not exist in our dictionary then we predict some random word
        
       preds=Counter(ngram_dict1[random.choice(list(uni_dict.keys()))]).most_common()
      
else:
        preds=Counter(uni_dict[word]).most_common()[:3]



if st.button("Predict"):
    #sugg=unipred(text)
    for i in preds:
            st.write(i[0:5])

    st.write('Done!')

