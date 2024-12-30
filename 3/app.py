from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, request, jsonify

app = Flask (__name__)

model = Ollama (model = 'llama3.2:1b')

prompt = ChatPromptTemplate.from_template (
    'Write an essay of 300 words on given {topic}'
)

parser = StrOutputParser ()

chain = prompt | model | parser

@app.route ('/', methods = ['POST'])
def genericFunction ():
    if request.method == 'POST':
        inputText = request.json['topic']
        response = chain.invoke ({'topic' : inputText})
        return jsonify ({
            'output' : response
        })
    return ''

if __name__ == '__main__':
    app.run (debug = True)