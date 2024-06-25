import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings



import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

import os

GROQ_API_KEY = os.getenv('GROQ_API_KEY')


st.set_page_config(page_title="Document Genie", layout="wide")

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Content:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGroq(model='llama3-70b-8192', groq_api_key=GROQ_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

chain = get_conversational_chain()


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



def user_input(user_question):
    # embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def main():
    st.header("Chat with your PDF")

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
            response = user_input(chat_history)
            print(response)
        with st.chat_message('assistant'):
            st.markdown(response['output_text'])
        st.session_state.messages.append({'role':'assistant', 'content':response['output_text']})

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button"):  # Check if API key is provided before processing
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()