#Conversation Q&A Chatbot
"""In many A&A applications we want to allow the user to have a back-and-forth conversation,meaning the application needs some sort of "memory" of pas questions and answers, and some logic for incorporating those into its current thinking.

In this guid we focus on adding logic for incorporating historical messages.

we will cover two approaches :
=>Chains, in which always execute a retrieval step;
=>Agents, in which we give an LLM decretion over wheter and how to execute a retrieval step(or multiple steps).
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
llm = ChatGroq(model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key)

from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings # instead of OPENAIEMBEDDINg
emebeddings= HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


#Web based loader
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs = dict(parse_only = bs4.SoupStrainer(
        class_=("post-title", "post-content","post-header") 
    ))
)
web_data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits,embedding=emebeddings)
retriever = vectorstore.as_retriever()

#Prompt Template
system_prompt = ("You are an assistant for question-answering tasks"
                 "Use the following pieces of retrieved context to answer"
                 "the question.If you dont know the answer , say that you"
                 "don't know. UUse three sentences maximum and keep the "
                 "answer concise"
                 "\n\n"
                 "{context}"
                 )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever,question_answer_chain)

response = rag_chain.invoke({"input":"What is Self Reflection"})
print(response['answer'])


#lets add chat history now

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history. Do Not answer the question"
    "just reformulate it if needed and otherwise return it as is"
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human","{input}")
])


history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

question_answer_chain = create_stuff_documents_chain


#extract the qachatbot and UNDERSTAND the whole code



