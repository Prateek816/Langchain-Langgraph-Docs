import os
from typing import List, Literal
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()

# Model for high-stakes reasoning (Grading/Routing)
reasoning_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# Embeddings for Vectorstore
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# ==========================================
# 2. VECTORSTORE INITIALIZATION
# ==========================================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# ==========================================
# 3. DATA MODELS & CHAINS
# ==========================================
class GradeDocuments(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance check: 'yes' or 'no'")

# Retrieval Grader Chain
grader_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the document is relevant to the question. Score 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grader_prompt | reasoning_llm.with_structured_output(GradeDocuments)

# RAG Generation Chain
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | reasoning_llm | StrOutputParser()

# Question Rewriter (optimized for web search)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "Transform the user's question into a version optimized for web search retrieval."),
    ("human", "Initial question: \n\n {question} \n Formulate an improved version."),
])
question_rewriter = rewrite_prompt | reasoning_llm | StrOutputParser()

# ==========================================
# 4. GRAPH STATE & NODES
# ==========================================
class GraphState(TypedDict):
    question: str
    generation: str
    run_web_search: str  # "Yes" or "No"
    documents: List[Document]

def retrieve(state: GraphState):
    print("---RETRIEVING FROM VECTORSTORE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents}

def grade_documents(state: GraphState):
    print("---CHECKING DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    
    filtered_docs = []
    run_web_search = "No"
    
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("---GRADE: RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: NOT RELEVANT---")
            run_web_search = "Yes"
            
    return {"documents": filtered_docs, "run_web_search": run_web_search}

def generate(state: GraphState):
    print("---GENERATING FINAL ANSWER---")
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": generation}

def transform_query(state: GraphState):
    print("---TRANSFORMING QUERY FOR WEB---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"question": better_question}

def web_search(state: GraphState):
    print("---EXECUTING WEB SEARCH---")
    docs = web_search_tool.invoke({"query": state["question"]})
    # Combine results into a single document object
    web_results = "\n".join([d["content"] for d in docs])
    new_doc = Document(page_content=web_results, metadata={"source": "tavily"})
    
    # Append to existing valid documents
    current_docs = state["documents"]
    current_docs.append(new_doc)
    return {"documents": current_docs}

# ==========================================
# 5. CONDITIONAL LOGIC & GRAPH ASSEMBLY
# ==========================================
def decide_to_generate(state: GraphState):
    if state["run_web_search"] == "Yes":
        print("---DECISION: NOT ALL DOCS RELEVANT, TRIGGERING WEB SEARCH---")
        return "transform_query"
    else:
        print("---DECISION: ALL DOCS RELEVANT, GENERATING---")
        return "generate"

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)

# Add Edges
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ==========================================
# 6. EXECUTION HELPER
# ==========================================
def run_rag(question: str):
    print(f"\nProcessing: {question}")
    inputs = {"question": question}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' completed.")
    
    # The 'generate' node is always the last node before END in this graph
    final_answer = output.get("generate", {}).get("generation", "No answer generated.")
    print(f"\nFINAL ANSWER:\n{final_answer}\n{'-'*50}")

# Test 1: Content in Vectorstore
run_rag("What are the types of agent memory?")

# Test 2: Content NOT in Vectorstore (Should trigger Web Search)
run_rag("How does the AlphaCodium paper work?")