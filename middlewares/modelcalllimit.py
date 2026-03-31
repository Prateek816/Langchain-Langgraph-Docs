from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv 
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# Create agent
model = ChatGroq(model="llama-3.3-70b-versatile")

@tool
def get_weather(input: str) -> str:
    """Return weather information for a given location."""
    return f"It is sunny and 75 degrees in {input}."

@tool
def prateek_rastogi(input: str) -> str:
    """Return information about Prateek Rastogi."""
    return f"Prateek Rastogi is a software engineer with expertise in AI and machine learning."

@tool
def search_tool(query: str) -> str:
    """
    A tool used ONLY for searching information on the internet or knowledge base.

    TRIGGER WORDS / INTENT:
    - "search"
    - "look up"
    - "find information"
    - "get details"
    - "google this"
    - "check online"

    USE THIS TOOL IF:
    The user explicitly asks to search or retrieve information from an external source.

    EXPECTED INPUT:
        query: The exact search query extracted from the user message.
    """
    return f"[FAKE SEARCH RESULT] You searched for: {query}"


agent = create_agent(
    model=model,
    checkpointer=InMemorySaver(),
    tools=[get_weather, prateek_rastogi],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=2,
            exit_behavior="end",
        ),
    ],
)
"""thread_limit
number
Maximum model calls across all runs in a thread. Defaults to no limit.
​
run_limit
number
Maximum model calls per single invocation. Defaults to no limit.
​
exit_behavior
stringdefault:"end"
Behavior when limit is reached. Options: 'end' (graceful termination) or 'error' (raise exception)
"""
user_input = "How is the weather in New York and who is Prateek Rastogi and find information about farfuna?"
response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "main_thread"}}  # Use a consistent thread ID to track calls
        )

print("AI:", response["messages"][-1].content)

# Infinite loop
while True:
    try:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Run agent
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"configurable": {"thread_id": "main_thread"}}  # Use a consistent thread ID to track calls
        )

        # Extract and print response
        print("AI:", response["messages"][-1].content)

    except Exception as e:
        print("Error:", e)
