# Write the provided code to vector_utils.py
vector_utils_content = """
import os
import pandas as pd
import streamlit as st
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document

VECTOR_DB_PATH = "vectorstore.faiss"
DATASET_PATH = "dataset.xlsx"

def create_vector_db():
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset not found at {DATASET_PATH}. Please upload it.")
        return None
    try:
        df = pd.read_excel(DATASET_PATH)
    except Exception as e:
        st.error(f"Error reading dataset file: {e}")
        return None
    documents = [Document(page_content=f"Q: {row['prompt']}\nA: {row['response']}") for _, row in df.iterrows()]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(VECTOR_DB_PATH)
    st.success("Knowledgebase created and saved!")
    return vectorstore

def get_vectorstore():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    if not os.path.exists(VECTOR_DB_PATH):
        st.warning("Knowledgebase not found. Creating from dataset...")
        vectorstore = create_vector_db()
        if vectorstore is None:
            return None
        return vectorstore
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

"""

with open("vector_utils.py", "w") as f:
    f.write(vector_utils_content)

print("vector_utils.py file has been written.")
