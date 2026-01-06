import os
import streamlit as st 
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_models import ChatOpenAI

#import pdf files
st.header("My First Chatbot")
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking a question",type="pdf")

# extract the files 
if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()
        #st.write(text)

    # Break it into chunks 
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size = 1000,
        chunk_overlap = 150,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # generating embedding 
    
    embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2")
    # creating vector store 
    vector_store = FAISS.from_texts(chunks , embeddings)

    #get user questions 
    user_question = st.text_input("Type your question here")

    #do similarity search 
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)

        # define LLM
        llm = ChatOpenAI(
        model="gpt-3.5-turbo",   
        temperature=0,            
        max_tokens=500)          
    
    #output result 
    llm = ChatOpenAI(model="gpt-3.5-turbo")

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based only on the context below:
    {context}

    Question: {question}
    """
)

    retriever = vector_store.as_retriever()

    chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    )

    response = chain.invoke(user_question)

    st.write(response.content)


    