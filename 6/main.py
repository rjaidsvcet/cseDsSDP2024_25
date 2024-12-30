import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv ()

groq_api_key = os.environ['GROQ_API_KEY']

if 'vector' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings (model = 'llama3')
    st.session_state.loader = WebBaseLoader ('https://docs.smith.langchain.com/')
    st.session_state.docs = st.session_state.loader.load ()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
    st.session_state.final_document = st.session_state.text_splitter.split_documents (st.session_state.docs)
    st.session_state.database = FAISS.from_documents (st.session_state.final_document, st.session_state.embeddings)

st.title ('Generic GROQ Application')
model = ChatGroq (groq_api_key=groq_api_key, model='llama3-8b-8192')

prompt = ChatPromptTemplate.from_template (
    '''
        Answer the questions based on provided context only.
        Please provide the most accurate response based on question.
        <context>
            {context}
        </context>
        Questions : {input}
    '''
)

document_chain = create_stuff_documents_chain (llm=model, prompt=prompt)
retriever = st.session_state.database.as_retriever ()
retriever_chain = create_retrieval_chain (retriever, document_chain)

inputPrompt = st.text_input ('Input your prompt here')

if inputPrompt:
    response = retriever_chain.invoke ({
        'input' : inputPrompt
    })
    st.write (response['answer'])