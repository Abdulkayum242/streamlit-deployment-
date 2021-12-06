import streamlit as st 
import pandas as pd
import numpy as np
import nltk 
import pickle
import time
import re
from nltk.tokenize import word_tokenize
from utils  import *


st.title('word prediction')
st.balloons()



def main_function(text): 
      while(True):
         cleaned_text=extra_space(text)
         cleaned_text=sp_charac(cleaned_text)
         tokenized=tokenize_text(cleaned_text)
         if len(tokenized)==1:
            ab=unipred(tokenized[0])
            
            break


text= st.text_input("Enter a sentence")

if st.button("Predict"):
    suggestion=main_function(text)
    st.write(suggestion)
    st.write('Done!')

