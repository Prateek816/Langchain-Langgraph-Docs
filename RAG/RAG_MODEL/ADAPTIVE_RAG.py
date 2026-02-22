import os
from pprint import pprint
from typing import List, Literal, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.schema import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, START
from pydantic import BaseModel, Field

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
load_dotenv()

def set_env_vars():
    keys = ["GOOGLE_API_KEY", "TAVILY_API_KEY", "GROQ_API_KEY", "LANGCHAIN_API_KEY"]
    for key in keys:
        os.environ[key] = os.getenv(key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

set_env_vars()

# Initialize LLM and Embeddings
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 2. DATA MODELS (Pydantic & TypedDict)
# ==========================================

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web search"] = Field(
        ..., description="Choose to route to web search or vectorstore."
    )

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Relevance check: 'yes' or 'no'")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination check."""
    binary_score: str = Field(description="Is the answer grounded in facts: 'yes' or 'no'")

class GradeAnswer(BaseModel):
    """Binary score to assess if answer addresses question."""
    binary_score: str = Field(description="Does the answer resolve the question: 'yes' or 'no'")

class GraphState(TypedDict):
    """Represents the state of our LangGraph."""
    question: str
    generation: str
    documents: List[Document]

# ==========================================
# 3. VECTORSTORE & TOOLS SETUP
# ==========================================

# Load and Split Documents
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)
doc_splits = text_splitter.split_documents(docs_list)

# Initialize Vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# Web Search Tool
web_search_tool = TavilySearchResults(k=3)

# ==========================================
# 4. CHAINS & GRADERS
# ==========================================

# Router Chain
router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at routing questions to vectorstore (agents, prompt engineering, attacks) or web search."),
    ("human", "{question}"),
])
question_router = router_prompt | llm.with_structured_output(RouteQuery)

# Retrieval Grader Chain
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "Assess if the retrieved document is relevant to the user question. Score 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
])
retrieval_grader = grade_prompt | llm.with_structured_output(GradeDocuments)

# RAG Generation Chain
rag_prompt = hub.pull("rlm/rag-prompt")
rag_chain = rag_prompt | llm | StrOutputParser()

# Hallucination Grader
hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "Is the generation grounded in the facts? Score 'yes' or 'no'."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
])
hallucination_grader = hallucination_prompt | llm.with_structured_output(GradeHallucinations)

# Answer Grader
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Does the answer resolve the question? Score 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
])
answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

# Question Rewriter
rewrite_system = "You are a question re-writer optimized for vectorstore retrieval. Improve the input question."
re_write_prompt = ChatPromptTemplate.from_messages([
    ("system", rewrite_system),
    ("human", "Initial question: \n\n {question} \n Formulate an improved version."),
])
question_rewriter = re_write_prompt | llm | StrOutputParser()

# ==========================================
# 5. GRAPH NODES (Functions)
# ==========================================

def retrieve(state: GraphState):
    print("---RETRIEVING---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def generate(state: GraphState):
    print("---GENERATING---")
    generation = rag_chain.invoke({"context": state["documents"], "question": state["question"]})
    return {"generation": generation}

def grade_documents(state: GraphState):
    print("---GRADING DOCUMENTS---")
    filtered_docs = []
    for d in state["documents"]:
        score = retrieval_grader.invoke({"question": state["question"], "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)
    return {"documents": filtered_docs}

def transform_query(state: GraphState):
    print("---TRANSFORMING QUERY---")
    better_question = question_rewriter.invoke({"question": state["question"]})
    return {"question": better_question}

def web_search(state: GraphState):
    print("---WEB SEARCHING---")
    docs = web_search_tool.invoke({"query": state["question"]})
    combined_content = "\n".join([d["content"] for d in docs])
    return {"documents": [Document(page_content=combined_content)]}

# ==========================================
# 6. CONDITIONAL EDGES (Logic)
# ==========================================

def route_question(state: GraphState):
    print("---ROUTING---")
    result = question_router.invoke({"question": state["question"]})
    return "web_search" if result.datasource == "web search" else "vectorstore"

def decide_to_generate(state: GraphState):
    print("---DECIDING NEXT STEP---")
    if not state["documents"]:
        return "transform_query"
    return "generate"

def grade_generation_v_documents_and_question(state: GraphState):
    print("---CHECKING HALLUCINATIONS & UTILITY---")
    # Check Hallucination
    score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    if score.binary_score == "yes":
        # Check if it actually answers the question
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        return "useful" if answer_score.binary_score == "yes" else "not useful"
    return "not supported"

# ==========================================
# 7. GRAPH CONSTRUCTION
# ==========================================

workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build Edges
workflow.add_conditional_edges(START, route_question, {"web_search": "web_search", "vectorstore": "retrieve"})
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", decide_to_generate, {"transform_query": "transform_query", "generate": "generate"})
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges("generate", grade_generation_v_documents_and_question, {
    "not supported": "generate",
    "useful": END,
    "not useful": "transform_query",
})

# Compile App
app = workflow.compile()

# ==========================================
# 8. EXECUTION
# ==========================================

def run_adaptive_rag(user_question: str):
    print(f"\n{'='*20}\nProcessing Question: {user_question}\n{'='*20}")
    inputs = {"question": user_question}
    final_output = None
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}' completed.")
            final_output = value
    
    if final_output and "generation" in final_output:
        print("\nFINAL ANSWER:")
        pprint(final_output["generation"])

# Run examples
run_adaptive_rag("Who will the Bears draft first in 2024?")
run_adaptive_rag("What are the types of agent memory?")