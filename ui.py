import streamlit as st

from rag import RAG
from llm import LLM
from chain import Chain
import threading

import os

model = LLM()
rag = RAG()

def upload(document_path):
    rag.chunk(document=document_path)
    rag.store()
    

def ask(query):
    chain = Chain(model=model, rag=rag)   
    response = chain.query(query)
    return response    

st.title('Chat with your Documents')

uploaded_file = st.file_uploader(label="Choose a file to chat with")

# Process the uploaded file
if uploaded_file is not None:
    file_path = f"./docs/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    upload(file_path)
    st.success("File uploaded and processed successfully!")


#Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
if prompt := st.chat_input():
    with st.chat_message('user'):
        st.markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    with st.spinner('Generating...'):
        chat_history = "\n".join([f"{history['role']}:{history['content']}" for history in st.session_state.messages])
        chat_history += prompt
        response = ask(chat_history)
    with st.chat_message('assistant'):
        st.markdown(response['answer'])
    st.session_state.messages.append({'role':'assistant', 'content':response['answer']})
    