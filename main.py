import numpy as np 
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

#Loading the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

#Loading the pretrained model 
model = load_model('simple_rnn_imdb.h5')

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen= 500)
    return padded_review

import streamlit as st

# Add some custom CSS to style the app
st.markdown("""
    <style>
        .title {
            font-size: 40px;
            font-weight: bold;
            color: #1e90ff;
            text-align: center;
        }
        .input-container {
            display: flex;
            justify-content: center;
            margin-bottom: 20px;
        }
        .input-textarea {
            width: 80%;
            height: 200px;
            padding: 15px;
            font-size: 18px;
            border-radius: 8px;
            border: 1px solid #ccc;
            resize: none;
        }
        .button {
            background-color: #1e90ff;
            color: white;
            font-size: 18px;
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
        }
        .button:hover {
            background-color: #4682b4;
        }
        .result {
            font-size: 24px;
            font-weight: bold;
            color: #008000;
            text-align: center;
        }
        .loading {
            font-size: 18px;
            color: #ffa500;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title of the application
st.markdown('<div class="title">Text Sentiment Analyser</div>', unsafe_allow_html=True)

# Provide instructions
st.write('Please enter the text below for sentiment analysis.')

# Text input from the user
user_input = st.text_area('Enter your text here', '', height=200, key='text_input')

# Add a button for classification
if st.button('Classify', key='classify_button', help="Click to analyze sentiment"):
    with st.spinner('Classifying...'):
        # Assuming `preprocess_text` and `model.predict` functions are already defined
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] >= 0.5 else 'Negative'
        probability = prediction[0][0] if sentiment == 'Positive' else 1 - prediction[0][0]

        # Display the sentiment result with enhanced UI
        st.markdown(f'<div class="result">Sentiment: {sentiment}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result">Probability: {probability}</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="loading">Please enter some text and click the "Classify" button to analyze sentiment.</div>', unsafe_allow_html=True)

