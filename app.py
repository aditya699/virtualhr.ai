'''
Author - Aditya Bhatt 7:00 AM 27-03-2024
AI Based Virtual HR
Flow
Speech -> Text -> text as prompt
PDF->Chunk->Embeddings(open ai)->retrival
text+retrival -> prompt in llm
Output->text>Speech
'''



import speech_recognition as sr
import pyttsx3
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage

# Initialize ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the document path
DOCUMENT_PATH = "https://docs.google.com/document/d/e/2PACX-1vS8uUx3jgheoS09Wo-JVCmuBuAbr_NeCk1TMlwHbEXbJnCoE287gdWs_Z33BIR_iSeD1aqQg4Y3BHil/pub"

# Initialize recognizer class (for recognizing the speech)
r = sr.Recognizer()

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define contextualization prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# Define question-answering prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Define a function to determine whether to use contextualization or not
def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

# Initialize an empty chat history
chat_history = []

# Run until user says "bye"
while True:
    # Reading Microphone as source
    with sr.Microphone() as source:
        print("I am Your Virtual HR Please Ask Your Query or say bye to end")
        engine.say("I am Your Virtual HR Please Ask Your Query or say bye to end")
        engine.runAndWait()
        audio_text = r.listen(source)
        print("thanks")
        engine.say("thanks")
        engine.runAndWait()

    # Recognize speech and convert it to text
    try:
        # Using Google Speech Recognition
        recognized_text = r.recognize_google(audio_text)
        print("Question: " + recognized_text)

        # Check if the user wants to end the conversation
        if recognized_text.lower() == "bye":
            print("Conversation ended.")
            break

        # Load, chunk, and index the contents of the predefined document.
        loader = WebBaseLoader(web_paths=(DOCUMENT_PATH,))
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

        # Retrieve and generate using the relevant snippets of the document.
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

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
        )

        # Generate answer
        answer = rag_chain.invoke({"question": recognized_text, "chat_history": chat_history})
        print("Answer:", answer)
        engine.say(answer)
        engine.runAndWait()

        # Add the user's question and AI's response to the chat history
        chat_history.extend([HumanMessage(content=recognized_text), HumanMessage(content=answer)])

    except sr.UnknownValueError:
        print("Sorry, I did not understand that")
        engine.say("Sorry, I did not understand that")
        engine.runAndWait()
    except sr.RequestError:
        print("Sorry, could not request results. Please check your internet connection")
        engine.say("Sorry, could not request results. Please check your internet connection")
        engine.runAndWait()
