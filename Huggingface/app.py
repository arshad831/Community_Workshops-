import streamlit as st
from transformers import pipeline

st.title("Sentiment Bot")

sentiment_pipeline = pipeline("sentiment-analysis")

user_input = st.text_area("Enter the Review")

if st.button("Analyze"):
    result = sentiment_pipeline(user_input)
    st.write("Sentiment", result[0]['label'])
    st.write("Confident Score ", result[0]['score'])

