import streamlit as st
import requests

def getOllamaResponse (inputText):
    response = requests.post (
        url = 'http://localhost:5000/',
        json = {'topic' : inputText}
    )
    return response.json ()['output']

st.title ('Llama 3 using Flask REST API')
inputText = st.text_input ('Enter the topic you want to write essay on')

if inputText:
    st.write (getOllamaResponse (inputText))