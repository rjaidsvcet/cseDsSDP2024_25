from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
# from langchain.chains import LLMChain
import streamlit as st

st.title ('Celebrity Information using Large Language Model')
inputText = st.text_input ('Search the celebrity you want')

inputPrompt = PromptTemplate (
    input_variables = ['name'],
    template = 'Tell me about the celebrity {name}'
)

llm = Ollama (model = 'llama3.2:1b')

# chain = LLMChain (
#     llm = llm,
#     prompt = inputPrompt,
#     verbose = True
# )

chain = inputPrompt | llm

# response = chain.invoke ({'name' : 'Marlon Brando'})
# print (response)

if inputText:
    st.write (chain.invoke ({
        inputText
    }))