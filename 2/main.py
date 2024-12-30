from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

prompt = ChatPromptTemplate.from_messages ([
    ('system', 'Respond to the queries as an assitant'),
    ('user', 'Question : {question}')
])

model = Ollama (model = 'llama3.2:1b')

parser = StrOutputParser ()

chain = prompt | model | parser

st.title ('Langchain API with Ollama LLM')
inputText = st.text_input ('Search the topic that you want')

if inputText:
    st.write (chain.invoke ({
        'question' : inputText
    }))