import streamlit as st
from langchain.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter

from rag import RAG
from llm import LLM
from chain import Chain

model = LLM()
rag = RAG()

#load the file here
rag.chunk(document='./')
rag.store()

chain = Chain(model = model.llm, rag=rag.langchain_retriever())

chain.invoke("Hi there")