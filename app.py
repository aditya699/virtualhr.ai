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
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env
load_dotenv()

# Access the API key
API_KEY = os.getenv("OPENAI_API_KEY")

# Define the Streamlit app
st.title("Create Your Own Company's Question Answer System")

st.image ("pic.png",width=500)
#question_input = st.text_input("Enter your Web Path for Company Policy:", "https://docs.google.com/document/d/e/2PACX-1vS8uUx3jgheoS09Wo-JVCmuBuAbr_NeCk1TMlwHbEXbJnCoE287gdWs_Z33BIR_iSeD1aqQg4Y3BHil/pub")
question_input ="https://docs.google.com/document/d/e/2PACX-1vS8uUx3jgheoS09Wo-JVCmuBuAbr_NeCk1TMlwHbEXbJnCoE287gdWs_Z33BIR_iSeD1aqQg4Y3BHil/pub"
# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=(question_input,),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
template = """
Use the following pieces of context to answer the question at the end as a Human Resource Manager.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Give all details based on context but keep the answer crisp

{context}

Question: {question}

Helpful Answer:
"""
custom_rag_prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "/n/n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
question_input = st.text_input("Enter your question:", "")
if question_input:
    answer = rag_chain.invoke(question_input)
    st.write("Answer:", answer)
