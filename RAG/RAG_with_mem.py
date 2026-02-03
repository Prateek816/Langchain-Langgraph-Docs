import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough,
    Runnable,
    RunnableBranch,
    RunnableWithMessageHistory,
)
from langchain_core.messages import AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Any, Dict


# ─────────────────────────────────────────
# LOAD KEYS
# ─────────────────────────────────────────
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


# ─────────────────────────────────────────
# LOAD DOCUMENTS + SPLITTING
# ─────────────────────────────────────────
loader = TextLoader("learning/test_resources/ai.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = splitter.split_documents(documents)


# ─────────────────────────────────────────
# EMBEDDINGS + VECTORSTORE + RETRIEVER
# ─────────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever()


# ─────────────────────────────────────────
# LLM (Gemini)
# ─────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    api_key=google_api_key
)


# ─────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────

# RAG prompt
rag_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Use the provided context and conversation history."
    ),
    HumanMessagePromptTemplate.from_template(
        "History:\n{history}\n\nContext:\n{context}\n\nQuestion: {question}\n"
    ),
])

# Direct LLM prompt (skip retrieval)
direct_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "Answer using your own knowledge and conversation history."
    ),
    HumanMessagePromptTemplate.from_template(
        "History:\n{history}\n\nQuestion: {question}\n"
    ),
])


# ─────────────────────────────────────────
# CHAINS
# ─────────────────────────────────────────

# RAG chain
rag_chain: Runnable[str, Any] = (
    {"context": retriever, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
    | rag_prompt
    | llm
)

# Direct chain
direct_chain: Runnable[str, Any] = (
    {"question": RunnablePassthrough(), "history": RunnablePassthrough()}
    | direct_prompt
    | llm
)

# Router: ok=True → direct, else → RAG
router_chain = RunnableBranch(
    (lambda d: d["ok"] is True, direct_chain),
    rag_chain
)


# ─────────────────────────────────────────
# MEMORY WRAPPER
# ─────────────────────────────────────────

memory_store: Dict[str, BaseChatMessageHistory] = {}

def get_memory(session_id: str):
    if session_id not in memory_store:
        memory_store[session_id] = BaseChatMessageHistory()
    return memory_store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    router_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="history",
)


# ─────────────────────────────────────────
# USAGE (CORRECT)
# ─────────────────────────────────────────

response = chain_with_memory.invoke(
    {"question": "What is AI?", "ok": False},
    config={"configurable": {"session_id": "user01"}},
)

print("\nANSWER:", response.content)
