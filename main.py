## Importing Packages
import pickle
import streamlit as st
from utils import *

@st.cache
def get_predictions(input_tokens, starts, k = 1.0):
    n_gram_counts_list = pickle.load(open('en_counts.txt', 'rb'))
    vocabulary = pickle.load(open('vocab.txt', 'rb'))
    suggestion = get_suggestions(input_tokens, n_gram_counts_list, vocabulary, k=k, start_with = starts)
    return suggestion

## Page Title
st.set_page_config(page_title = "SwiftKey Text Prediction",
    page_icon = "💬")
st.title("SwiftKey Text Prediction")
st.markdown("---")

## Sidebar
st.sidebar.header("The Capstone project is done in collaboration with Swiftkey.")
st.sidebar.markdown("---")
st.sidebar.markdown("People are spending an increasing amount of time on their mobile devices.But typing on mobile devices can be a difficult task.SwiftKey builds a smart keyboard that makes it easier for people to type on their mobile devices. One cornerstone of their smart keyboard is predictive text models. When someone types you, the keyboard presents option for what the next word might be. For example, the word might be are.")

## Input Fields
sentence = st.text_input("Enter a sentence")
st.subheader("Optional Inputs")
starts = st.text_input("The starting letter of the expected next word")
k = st.number_input("Enter smoothing factor k")

tokenized = sentence.split()

if st.button("Predict"):
    suggestion = get_predictions(tokenized, starts, k)
    st.write(suggestion[0][1][0])
        
    


st.markdown("---")
st.markdown("All Rights Reserved.")


