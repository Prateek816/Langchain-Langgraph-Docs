import os
import pprint
from typing import Annotated, Literal, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_classic import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from langchain_classic.tools.retriever import create_retriever_tool
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ==========================================
# 1. ENVIRONMENT & MODEL SETUP
# ==========================================
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

# Using Llama-3.3-70b for high-reliability tool calling and grading
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ==========================================
# 2. VECTORSTORE & RETRIEVER TOOL
# ==========================================
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Splitting documents for the vectorstore
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embeddings,
)
retriever = vectorstore.as_retriever()

# Creating the tool that the agent will "decide" to use
retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search information about LLM agents, prompt engineering, and adversarial attacks.",
)
tools = [retriever_tool]

# ==========================================
# 3. STATE & DATA MODELS
# ==========================================
class AgentState(TypedDict):
    """The state of the graph, holding the message history."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

class RelevanceGrade(BaseModel):
    """Binary score for relevance check."""
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# ==========================================
# 4. GRAPH NODES (Logic Units)
# ==========================================

def agent(state: AgentState):
    """Decides if it needs to use a tool or just answer."""
    print("---CALLING AGENT---")
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def retrieve_docs(state: AgentState):
    """Executes the tool call to get documents."""
    # This matches the tool node logic
    tool_node = ToolNode(tools)
    return tool_node.invoke(state)

def generate_answer(state: AgentState):
    """Produces the final RAG response."""
    print("---GENERATING RESPONSE---")
    question = state["messages"][0].content
    context = state["messages"][-1].content # Content from the retriever
    
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = prompt | llm | StrOutputParser()
    
    response = rag_chain.invoke({"context": context, "question": question})
    return {"messages": [HumanMessage(content=response)]}

def rewrite_query(state: AgentState):
    """Re-phrases the user question for better retrieval."""
    print("---TRANSFORMING QUERY---")
    question = state["messages"][0].content
    
    prompt = f"Reason about the semantic intent of this question and improve it: {question}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# ==========================================
# 5. CONDITIONAL EDGES (Decision Logic)
# ==========================================

def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Grades the relevance of the retrieved data."""
    print("---CHECKING RELEVANCE---")
    
    # Define grader chain
    llm_grader = llm.with_structured_output(RelevanceGrade)
    prompt = PromptTemplate(
        template="""Assess if the document is relevant to the question. 
        Document: {context}
        Question: {question}
        Return 'yes' or 'no'.""",
        input_variables=["context", "question"],
    )
    grader_chain = prompt | llm_grader

    question = state["messages"][0].content
    docs = state["messages"][-1].content
    
    scored_result = grader_chain.invoke({"question": question, "context": docs})
    
    if scored_result.binary_score == "yes":
        print("---DECISION: RELEVANT---")
        return "generate"
    else:
        print("---DECISION: NOT RELEVANT---")
        return "rewrite"

# ==========================================
# 6. GRAPH CONSTRUCTION
# ==========================================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode(tools))
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("generate", generate_answer)

# Set up edges
workflow.add_edge(START, "agent")

# Agent decision: Tool or Finish?
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# After retrieval: Relevant or Rewrite?
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate": "generate",
        "rewrite": "rewrite"
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

# Compile
graph = workflow.compile()

# ==========================================
# 7. EXECUTION
# ==========================================
inputs = {
    "messages": [
        ("user", "What does Lilian Weng say about the types of agent memory?"),
    ]
}

for output in graph.stream(inputs):
    for key, value in output.items():
        print(f"\n--- Node '{key}' completed ---")
        pprint.pprint(value, indent=2, width=80)