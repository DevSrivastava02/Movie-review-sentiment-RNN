import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

# Mapping the word index back to word
word_index=imdb.get_word_index()
word_index
reverse_word_index={values:key for key ,values in word_index.items()}

model=load_model('model.h5')

# Helper function
def decoded_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in decoded_review])


# Function to preprocess user input
def preprocess_text(text):
    words=text.lower().split()
    encoded_reivew=[word_index.get(word,2)+3 for word in words]
    padded_review=pad_sequences([encoded_reivew],maxlen=500)
    return padded_review

# Prediction fuinction
def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment,prediction[0][0]

# Streamlit
st.title('IMDB MOVIE REVIEW SENTIMENT ANALYSIS')
st.write('Enter a movie review to classify it as postive or negative.')

# User input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocesed_input=preprocess_text(user_input)

    # Make prediction
    prediction=model.predict(preprocesed_input)
    sentiment='Positive' if prediction[0][0]> 0.5 else 'Negative'


    # Display the result

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score : {prediction[0][0]}')

else:
    st.write('Please enter a movie review')
    
