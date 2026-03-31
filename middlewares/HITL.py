from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")
# ============================================================
# SEARCH TOOL
# ============================================================
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


# ============================================================
# SEND EMAIL TOOL
# ============================================================
@tool("send_email")
def send_email_tool(content: str) -> str:
    """
    A tool for sending an email. 
    This tool MUST be used when the user requests any action involving sending an email.

    TRIGGER WORDS / INTENT:
    - "send an email"
    - "email the team"
    - "send mail"
    - "notify via email"
    - "mail them"
    - "email X"

    USE THIS TOOL IF:
    The user directly requests sending an email, regardless of whether the content
    is fully specified or needs to be drafted by the agent.

    EXPECTED INPUT:
        content: The full email text prepared by the agent.
    """
    return f"[EMAIL BLOCKED] Email content was: {content}"


# ============================================================
# DELETE DATABASE TOOL
# ============================================================
@tool("delete_database")
def delete_database_tool(target: str) -> str:
    """
    A destructive tool used ONLY for deleting databases or wiping data.

    TRIGGER WORDS / INTENT:
    - "delete the database"
    - "wipe all data"
    - "remove the database"
    - "clear everything"
    - "drop the database"

    USE THIS TOOL IF:
    The user clearly expresses destructive intent toward stored data.

    EXPECTED INPUT:
        target: The name of the database or dataset to delete.
    """
    return f"[BLOCKED] Attempted to delete database: {target}"
SYSTEM_PROMPT = """You are a high-precision Gmail Operations Agent. 

### CORE CAPABILITIES:
1. **Gmail Management:** You can search, read, and draft emails.
2. **Context Persistence:** You have a sliding-window short-term memory and a long-term RAG archive. 

### OPERATIONAL RULES:
- **The ID Rule:** When a user asks about an email, ALWAYS use `search_gmail` first. Extract the exact 'id' from the tool output. NEVER hallucinate or guess an ID.
- **The Archive Rule:** If a user refers to a detail from a conversation that happened "a while ago" and it isn't in your current message history, use the `search_archive` tool to retrieve it from your RAG layer.
- **Precision:** When summarizing emails, focus on: Sender, Date, Subject, and Key Action Items.

### LOGIC FLOW:
1. **Search Phase:** Find the relevant email thread.
2. **Verification Phase:** Ensure the ID exists in the tool output.
3. **Action Phase:** Retrieve the body or draft the reply using that specific ID.
4. **Memory Phase:** If you find a critical piece of info (like a flight time or a meeting link), explicitly mention it so the system can archive it for later.

### CONSTRAINTS:
- Do not send emails unless the user explicitly uses the word "Send". Otherwise, always create a "Draft".
- If a tool returns an error, explain the error to the user rather than retrying indefinitely.
- You are communicating with Prateek. Keep responses professional, concise, and technical."""
agent = create_agent(
    model = llm,
    tools=[search_tool, send_email_tool, delete_database_tool],
    middleware = [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email":True,
                "delete_database":True,
                #Auto approve for safe operations
                "search":False,
            }
        )
    ],
    #To use Human In The Loop, we need to use a checkpoint saver to save the state of the agent and resume after human approval
    checkpointer=InMemorySaver(),
    system_prompt=SYSTEM_PROMPT
)

config = {"configurable":{"thread_id":"thread1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Send an email to the team"}]},
    config=config
)
result = agent.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config  # Same thread ID to resume the paused conversation
)
print(result['messages'])