
#INSTALLS 
# Install the OpenAI package for access to OpenAI's language models
#pip install openai
# Install Langchain, which is essential for natural language processing tasks in our chatbot
#pip install langchain
# Install Streamlit, a user-friendly tool for creating web-based chatbot interfaces
#pip install streamlit
# Install Streamlit-Chat, an extension for Streamlit that enhances chatbot functionality
#pip install streamlit-chat
# Install Faiss-CPU, a library for efficient similarity search and clustering (useful for chatbot operations)
#pip install faiss-cpu
# Install PyPDF to enable our chatbot to work with PDF documents
#pip install pypdf
# Install Python-Dotenv for managing environment variables securely
#pip install python-dotenv
# Install Tiktoken, a useful tool for tokenizing text data
#pip install tiktoken


# IMPORTS 

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
#from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
#from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai 
import json
import yaml
import os
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader



# GET  FILE AND Create VECTOR DB and Return the Vector DB 
def get_faiss_vectordb(file: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)
    

    embedding = OpenAIEmbeddings()
    
    # Create a unique FAISS index path based on the input file's name.
    faiss_index_path = f"faiss_index_{filename}"

    # Determine the loader based on the file extension.
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    else:
        # If the document type is not supported, print a message and return None.
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create a FAISS vector database from the chunked documents and embeddings.
    vectordb = FAISS.from_documents(doc_chunked, embedding)
    
    # Save the FAISS vector database locally using the generated index path.
    vectordb.save_local(faiss_index_path)
    
    # Return the FAISS vector database.
    return vectordb

# Create a Retriever On the Vector DB
def run_llm(vectordb, query: str) -> str:
    # Create an instance of the ChatOpenAI with specified settings.
    openai_llm = ChatOpenAI(temperature=0, verbose=True)
    
    # Create a RetrievalQA instance from a chain type with a specified retriever.
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=openai_llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    
    # Run a query using the RetrievalQA instance.
    answer = retrieval_qa.run(query)
    
    # Return the answer obtained from the query.
    return answer


#--------------# Main function #------------------------------------------------------

def main():


#GET API KEY 
# Set the page  for the Streamlit app.

    # Load environment variables from .env file
    load_dotenv(dotenv_path="api.env")

    # Now you can access your API key as an environment variable
    api_key = os.getenv('OPENAI_API_KEY')

    st.title("📝 File GPT using Open AI GPT 3.5 model")

                      
# Allow the user to upload a file with supported extensions.
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))

# Provide a text input field for the user to ask questions about the uploaded article.
    question = st.text_input(
           "Ask something about the article",
            placeholder="Ask something about the Budget 2024",
            disabled=not uploaded_file,
             )
                    
# If an uploaded file is available, process it.
    if uploaded_file:
    # Save the uploaded file locally.
       with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    # Create a FAISS vector database from the uploaded file.
       vectordb = get_faiss_vectordb(uploaded_file.name)
    
    # If the vector database is not created (unsupported file type), display an error message.
       if vectordb is None:
          st.error(
          f"The {uploaded_file.type} is not supported. Please load a file in pdf, txt, or md"
          )

    # Display a spinner while generating a response.
    with st.spinner("Generating response..."):
    # If both an uploaded file and a question are available, run the model to get an answer.
      if uploaded_file and question:
         answer = run_llm(vectordb=vectordb, query=question)
        # Display the answer in a Markdown header format.
         st.write("### Answer")
         st.write(f"{answer}")

if __name__ == '__main__':
    main()




