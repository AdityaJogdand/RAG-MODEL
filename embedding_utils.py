import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os

CHROMA_DIR = "chroma_db"

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def get_text_chunks(text, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents([text])

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def save_to_chroma(docs):
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def load_chroma():
    embeddings = get_embeddings()
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
