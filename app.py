from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain import HuggingFaceHub
import os
import langchain
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import re

import google.generativeai as genai
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import urllib
import warnings
from pathlib import Path as p
from pprint import pprint
from langchain.vectorstores import Chroma
import pandas as pd
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA



## Streamlit App
st.title("DataBot: Data Answer Tool 📈")
st.sidebar.title("New Documents ")
process_document_clicked = st.sidebar.button("Process Documents")
main_placeholder = st.empty()

## Load API Key
load_dotenv("./keys.env")
llm_api_key = os.getenv('Gemini_key')
os.environ["Gemini_key"] = llm_api_key

## User can   Load Document
document = None
for i in range(1):
    document = st.sidebar.file_uploader("Upload a PDF file", type="pdf")


## Load Model
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest",google_api_key=llm_api_key,
                             temperature=0.2,convert_system_message_to_human=True)

if process_document_clicked:
    temp_file = os.path.join('/tmp', document.name)
    with open(temp_file, 'wb') as f:
        f.write(document.getvalue())
        file_name = document.name



    pdf_loader = PyPDFLoader(document_path)

    main_placeholder.text("Data Loading...Started...✅✅✅")
    pages = pdf_loader.load_and_split()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    context = "\n\n".join(str(p.page_content) for p in pages)
    

    main_placeholder.text("Text Splitter...Started...✅✅✅")
    texts = text_splitter.split_text(context)

    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F" # emoticons
        u"\U0001F300-\U0001F5FF" # symbols & pictographs
        u"\U0001F680-\U0001F6FF" # transport & map symbols
        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", 
        flags=re.UNICODE
    )
    for i in range(len(texts)):
        texts[i] = emoji_pattern.sub(r'', texts[i])
    # create embeddings and save it to FAISS index
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=llm_api_key)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    vector_index = vectorstores.Chroma.from_texts(texts, embeddings)

    retriever = vector_index.as_retriever(search_kwargs={"k": 5})
    if hasattr(vectorstores.Chroma, "persist"):  # LangChain v0.1.16 or later
        retriever.persist(persist_path)
    else:   # Older LangChain versions
        vector_index.persist(persist_path)

    # time.sleep(2)



query = main_placeholder.text_input("Question: ")
if query:
    

    file_path = "./Database"

    if os.path.exists(file_path):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=llm_api_key)
        vectordb = Chroma(persist_directory="./Database", embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5})


        chain = RetrievalQA.from_chain_type(
                    model,
                    retriever=retriever,
                    return_source_documents=False,
                    chain_type="stuff"

        )
        result = chain({"query": query},return_only_outputs=True)
        # result will be a dictionary of this format --> {"answer": "", "sources": [] }
        st.header("Answer: ")
        st.write(result["result"])