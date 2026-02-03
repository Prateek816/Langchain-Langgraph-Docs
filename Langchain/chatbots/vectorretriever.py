#Vector Store and retrievers
"""This video tutorial will familiarize you with Langchain's vector store and retriever abstractions. These abstractions are designed to 
support retrieval of data--from (vector) databases and other sources -- for integration with LLM workflows . They are important for applications that fetcch data to be 
reasoned over as part of model inference , as in the case of RAG

We will cover
=>Documents
=>Vector Stores
=>Retrievers
"""

# Documents (MUST READ)
"""Langchain implements a Document abstrations , which is intended to represent a unit of text and associated metadat.
It has two attributes:

=>page_content : a string representing the content;
=>metadata : a dict containing arbitrary metadata. The metadata atrribute can capure information about t
he source of document, its relationship to other documents, and other information.Note that an individual Document object often represents a chunk of a larger documnent

"""

from langchain_core.documents import Document
documents = [
    Document(
        page_content="Dogs are great companion",
        metadata = {"source":"mamal-pets-doc"}
    ) 
]

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = ChatGroq(groq_api_key=groq_api_key,model = "openai/gpt-oss-120b")

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#print(embedding)

from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents,embedding=embedding)

print(vectorstore.similarity_search("cat"))

 







