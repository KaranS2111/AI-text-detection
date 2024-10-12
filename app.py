import streamlit as st
import re
import pickle
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') 
nltk.download('punkt') 
stop_words = set(stopwords.words('english'))
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
porter = PorterStemmer()

# Functions for preprocessing
def remove_stop_words(sentence): 
    words = sentence.split() 
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def remove_html(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_punct(data):
    return data.translate(str.maketrans('', '', string.punctuation))

def remove_url(data):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', data)

def stem_sentence(sentence):
    tokens = word_tokenize(sentence)
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def preprocess(data):
    data = data.lower()
    data = remove_html(data)
    data = remove_url(data)
    data = remove_punct(data)
    data = remove_stop_words(data)
    data = stem_sentence(data)
    return data

# Load the pre-trained model and vectorizer
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
model_new = pickle.load(open('model_GaussianNB.pkl', 'rb'))

# CSS for minimalistic and colorful design
st.markdown("""
    <style>
    body {
        background-color: #f4f4f9;
    }
    .reportview-container {
        background: #f4f4f9;
    }
    .css-1offfwp {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    .stButton button {
        background-color: #0072ff;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1);
    }
    .stButton button:hover {
        background-color: #005ec2;
    }
    .stSubheader {
        font-size: 20px;
        color: #0072ff;
        margin-bottom: 10px;
        text-align: center;
    }
    .stCaption {
        color: #6c757d;
        font-style: italic;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("🔍 AI Text Detector")
st.markdown("<div class='css-1offfwp'><h1>Is your text AI or Human-generated?</h1></div>", unsafe_allow_html=True)

# Input box for user text
input_sms = st.text_area("Enter the Text", placeholder="Type or paste text here...", height=150)

# Predict Button
if st.button('Predict'):

    # Preprocessing
    transformed_sms = preprocess(input_sms)

    # Vectorizing the input
    vector_input = tfidf.transform([transformed_sms])

    # Predict probabilities
    result = model_new.predict_proba(vector_input)[0]
    human = round(result[0] * 100, 2)
    ai = round(result[1] * 100, 2)

    # Display Results
    st.caption("Note: This is an experimental tool and may not always provide perfect results.")
    st.caption("Probabilities are shown below:")
    
    st.markdown(f"<div style='text-align: center;'><h3 style='color: #0072ff;'>🤖 AI: {ai}%</h3></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center;'><h3 style='color: #28a745;'>👤 Human: {human}%</h3></div>", unsafe_allow_html=True)
