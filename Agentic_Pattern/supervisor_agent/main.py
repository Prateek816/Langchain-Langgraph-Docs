import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
# LangChain v1.0 Standard Agent Factory
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END

load_dotenv()

# 1. LLM Setup
# NOTE: llama-3.3-70b-versatile is much more stable for tool-calling than the 8b version.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# 2. Worker Tools
@tool
def search_web(query: str) -> str:
    """Search the web for up-to-date information."""
    return f"[Search Results for '{query}']: LangChain v1.0 standardizes agent building via create_agent."

@tool
def calculate(expression: str) -> str:
    """Evaluate mathematical expressions."""
    try:
        # For production, use a safe evaluator like simpleeval
        return f"Result: {eval(expression, {'__builtins__': {}})}"
    except Exception as e:
        return f"Error: {e}"

@tool
def run_python_code(code: str) -> str:
    """Execute Python code and return output."""
    import io, contextlib
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, {})
        return buffer.getvalue() or "(Success)"
    except Exception as e:
        return f"Error: {e}"

# 3. Specialist Worker Agents
def make_worker_graph(name: str, objective: str, tools: list):
    """
    create_agent is the v1.0 standard. 
    It handles the tool-calling loop and returns a compiled graph.
    """
    # CRITICAL FIX for Groq: Instructions to prevent hallucinating 'brave_search'
    system_prompt = (
        f"You are the {name}. {objective}\n"
        "STRICT RULE: You only have access to the provided tools. "
        "Do NOT attempt to use 'brave_search', 'wolfram_alpha', or any tool not listed. "
        "If you need to search, use ONLY 'search_web'."
    )
    
    return create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt
    )

# Pre-compile specialist graphs
RESEARCH_GRAPH = make_worker_graph("Research Agent", "Find facts using web search.", [search_web])
MATH_GRAPH = make_worker_graph("Math Agent", "Solve math and logic problems.", [calculate])
CODE_GRAPH = make_worker_graph("Code Agent", "Write and execute Python snippets.", [run_python_code])

# 4. Supervisor State
class SupervisorState(TypedDict):
    # Annotated handles message accumulation automatically in LangGraph
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    next: str
    worker_outputs: dict

# 5. Supervisor Node
def supervisor_node(state: SupervisorState):
    system_msg = (
        "You are a Supervisor. Route the task to: research_agent, math_agent, or code_agent.\n"
        "If the task is complete, respond with FINISH.\n"
        "Respond ONLY with the name of the agent or FINISH."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Worker Progress: {worker_outputs}. What is the next step?"),
    ])
    
    chain = prompt | llm
    response = chain.invoke(state)
    decision = response.content.strip().replace(".", "").lower()
    
    # Validation
    valid_agents = ["research_agent", "math_agent", "code_agent"]
    if decision not in valid_agents:
        decision = "FINISH"
        
    return {"next": decision}

# 6. Worker Node Integration
def call_worker(state: SupervisorState, name: str, worker_graph):
    # In v1.0, agents accept a 'messages' key in their input dict
    result = worker_graph.invoke({"messages": state["messages"]})
    last_ai_msg = result["messages"][-1]
    
    return {
        "messages": [last_ai_msg],
        "worker_outputs": {**state.get("worker_outputs", {}), name: last_ai_msg.content}
    }

# 7. Final Answer Synthesis
def synthesis_node(state: SupervisorState):
    prompt = "Based on these worker results, provide a final polished answer: {outputs}"
    chain = ChatPromptTemplate.from_template(prompt) | llm
    response = chain.invoke({"outputs": state["worker_outputs"]})
    return {"messages": [AIMessage(content=response.content)]}

# 8. Main Graph Construction
def build_master_graph():
    builder = StateGraph(SupervisorState)
    
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("synthesis", synthesis_node)
    
    # Map the workers to the calling function
    builder.add_node("research_agent", lambda s: call_worker(s, "research_agent", RESEARCH_GRAPH))
    builder.add_node("math_agent", lambda s: call_worker(s, "math_agent", MATH_GRAPH))
    builder.add_node("code_agent", lambda s: call_worker(s, "code_agent", CODE_GRAPH))

    builder.add_edge(START, "supervisor")
    
    builder.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "research_agent": "research_agent",
            "math_agent": "math_agent",
            "code_agent": "code_agent",
            "FINISH": "synthesis"
        }
    )

    # All workers loop back to the supervisor
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("math_agent", "supervisor")
    builder.add_edge("code_agent", "supervisor")
    
    builder.add_edge("synthesis", END)
    return builder.compile()

# 9. Run the System
if __name__ == "__main__":
    master_app = build_master_graph()
    user_query = "Hello"
    
    final_state = master_app.invoke({
        "messages": [HumanMessage(content=user_query)],
        "worker_outputs": {}
    })
    
    print("\n" + "="*30)
    print("FINAL ANSWER:")
    print(final_state["messages"][-1].content)
    print("="*30)