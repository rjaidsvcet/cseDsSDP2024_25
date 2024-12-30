import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time

load_dotenv ()

groq_api_key = os.getenv ('GROQ_API_KEY')

st.title ('LLAMA3 with GROQ')

llm = ChatGroq (
    groq_api_key = groq_api_key,
    model = 'llama3-8b-8192'
)

prompt = ChatPromptTemplate.from_template (
    '''
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
            {context}
        </context>
        Questions : {input}
    '''
)

def vector_embedding ():
    st.session_state.embeddings = OllamaEmbeddings (model = 'llama3')
    st.session_state.loader = PyPDFDirectoryLoader ('./pdfFiles')
    st.session_state.docs = st.session_state.loader.load ()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter (chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents (st.session_state.docs[:3])
    st.session_state.vectors = FAISS.from_documents (st.session_state.final_documents, st.session_state.embeddings)

input_prompt = st.text_input ('Enter your questions from document')

if st.button ('Document Embedding'):
    vector_embedding ()
    st.write ('Vector Database is ready')

if input_prompt:
    document_chain = create_stuff_documents_chain (llm, prompt)
    retriever = st.session_state.vectors.as_retriever ()
    retriever_chain = create_retrieval_chain (retriever, document_chain)
    start = time.process_time ()
    response = retriever_chain.invoke ({
        'input' : input_prompt
    })
    print (f'Response Time : {time.process_time () - start}')
    st.write (response['answer'])