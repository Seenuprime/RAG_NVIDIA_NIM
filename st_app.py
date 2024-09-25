import streamlit as st 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["NVIDIA_API_KEY"] = os.getenv('NVIDIA_API_KEY')

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding():
    if 'vectors' not in st.session_state:
        st.session_state.embeddings = NVIDIAEmbeddings()
        st.session_state.docs = PyPDFDirectoryLoader("./data").load()
        st.session_state.splitted_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300).split_documents(st.session_state.docs[:30])
        st.session_state.vector_docs = FAISS.from_documents(st.session_state.splitted_docs, st.session_state.embeddings)


st.title("Nvidia NIM Project")

prompt = ChatPromptTemplate.from_template(
    """Answer the following question based on the given context,
    provide most accurate based on the que  stion
    <context>
    {context}
    </context>
    question: {input}
    """
)

query = st.text_input("Enter the query: ")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Your Vectore Store DB is ready")

if query:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vector_docs.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": query}) 
    st.success(response['answer'])