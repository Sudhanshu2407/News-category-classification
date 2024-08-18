import streamlit as st
import pickle
import numpy as np

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Title of the app
st.title('News Category Classifier')

# Subtitle
st.write('Enter a news headline to predict its category.')

# Input field for news headline
headline = st.text_input('News Headline')

# Predict button
if st.button('Predict'):
    if headline:
        # Preprocess the input
        input_data = vectorizer.transform([headline])
        
        # Make the prediction
        prediction = model.predict(input_data)[0]
        
        # Display the result
        st.success(f'The predicted category is: **{prediction}**')
    else:
        st.error('Please enter a news headline.')

# Footer
st.markdown('---')
st.write('This app classifies news headlines into predefined categories using a trained Naive Bayes model.')

