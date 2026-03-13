import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()

@tool
def send_email(recipient: str, subject: str, body: str) -> str:
    """Send an email to a recipient."""
    return f"Email sent successfully to {recipient}"

# 1. Use a valid Groq model identifier
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# 2. Setup Middleware
# True enables Approve, Edit, and Reject for that tool
hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={"send_email": True},
    description_prefix="Email pending approval"
)

# 3. Create the Agent
agent = create_agent(
    model=llm,
    tools=[send_email],
    middleware=[hitl_middleware],
    checkpointer=InMemorySaver(), # Required for middleware interrupts
    system_prompt="You are a helpful assistant for Sydney."
)

# 4. Running the Agent (Handling the Interrupt)
config = {"configurable": {"thread_id": str(uuid.uuid4())}}
input_data = {"messages": [("user", "Send a budget email to partner@startup.com")]}

# The execution will pause when it hits the middleware
result = agent.invoke(input_data, config)

# 5. The "Human" Part: Checking for the Interrupt
if result.get("interrupts"):
    print("\n--- AGENT PAUSED: ACTION REQUIRED ---")
    # In a real app, you'd show this to the user in a UI
    
    # Example: Implementing your 'Edit' decision
    # We use Command(resume=...) to talk back to the middleware
    decision = {
        "decisions": [{
            "type": "edit",
            "edited_action": {
                "name": "send_email",
                "args": {
                    "recipient": "partner@startup.com",
                    "subject": "RE: Budget",
                    "body": "I can only approve $500k."
                }
            }
        }]
    }
    
    # Resume the agent with the human decision
    final_result = agent.invoke(Command(resume=decision), config)
    print(final_result["messages"][-1].content)