"""
Documents -> Chunks -> Semantic Retrival(With MMR) + Keyword Retrieval -> ReRanking(FlashRanker) -> final relevant chunks

"""


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_classic.schema import Document


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
] #Testing Resources
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

docs_list = [
    Document(page_content="LangChain helps build LLM applications."),
    Document(page_content="Pinecone is a vector database for semantic search."),
    Document(page_content="The Eiffel Tower is located in Paris."),
    Document(page_content="Langchain can be used to develop agentic ai application."),
    Document(page_content="Langchain has many types of retrievers.")
]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
dense_vectorstore = FAISS.from_documents(doc_splits, embedding_model)
dense_retriever = dense_vectorstore.as_retriever(search_type="mmr",search_kwargs={"k": 3})


sparse_retriever = BM25Retriever.from_documents(doc_splits)
sparse_retriever.k = 3

hybrid_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3],
    verbose=True
)


import pprint

query ="How can I build an app using LLMs?"
results = hybrid_retriever.invoke(query)

for i,docs in enumerate(results):
    print(f"\n Document {i+1}: {docs.page_content}")

from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker

flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
compressor = FlashrankRerank(client=flashrank_client)

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=hybrid_retriever
)
query = "How can I build an app using LLMs?"
compressed_docs = compression_retriever.invoke(query)

print("\n--- Reranked Documents ---")
for i, doc in enumerate(compressed_docs):
    score = doc.metadata.get("relevance_score", "N/A")
    print(f"Rank {i+1} (Score: {score:.8f}): {doc.page_content}")

