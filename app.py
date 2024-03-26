'''
Author    -Aditya Bhatt 19:00 PM V1
Objective -Bulid a basic rag qna that can communicate with company internal Data
Data -> Split -> Store (Chroma DB)-> Retrival(Question + Search) in LLM(gpt3.5 turbo)
Comments-
1.Basic RAG System to communicate with your company internal Data is setup.
(Note this is a simple QA System)
'''

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

# Load environment variables from .env
load_dotenv()

# Access the API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Define the Streamlit app
st.title("Ask Questions about Company Policies and Procedures")

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://docs.google.com/document/d/e/2PACX-1vS8uUx3jgheoS09Wo-JVCmuBuAbr_NeCk1TMlwHbEXbJnCoE287gdWs_Z33BIR_iSeD1aqQg4Y3BHil/pub",),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Example messages for prompt invocation
example_messages = prompt.invoke(
    {"context": "filler context", "question": "filler question"}
).to_messages()

# Get user question input
user_question = st.text_input("Ask your question about company policies and procedures:")

# Display output based on user question
if user_question:
    st.write("Answer:")
    # Join the output into a single sentence
    answer = ' '.join(chunk for chunk in rag_chain.stream(user_question))
    st.write(answer)
