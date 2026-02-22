import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# The modern, stable equivalent for tool-use and reasoning
llm = ChatGroq(model="llama-3.3-70b-versatile")
from pprint import pprint
#pprint(llm.invoke("HI").content)

message = [
    {
        "role":"system","content":"You are a helpful assistant",
        "role":"user","content":"Hi how are you"
    }
]
#pprint(llm.invoke(message).content)

import os
from typing import Annotated, TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # Use SqliteSaver for file persistence

# 1. Define the State (this replaces your manual self.message list)
class ChatState(TypedDict):
    # add_messages ensures new messages are appended rather than overwriting
    messages: Annotated[list, add_messages]

class Chatbot:
    def __init__(self, system=""):
        self.llm = ChatGroq(model="llama-3.1-8b-instant")
        self.system = system
        
        # 2. Set up the Graph
        workflow = StateGraph(ChatState)
        workflow.add_node("model", self._call_llm)
        workflow.add_edge(START, "model")
        workflow.add_edge("model", END)
        
        # 3. Add Persistence (The "Checkpointer")
        # Use MemorySaver() for RAM-only or SqliteSaver for disk
        self.memory = MemorySaver()
        self.app = workflow.compile(checkpointer=self.memory)
        
        # 4. Thread Configuration (Identifies this specific conversation)
        self.config = {"configurable": {"thread_id": "default_user"}}

    def _call_llm(self, state: ChatState):
        # Inject system message if it's a new conversation
        messages = state["messages"]
        if self.system and not any(m.type == 'system' for m in messages):
            messages = [{"role": "system", "content": self.system}] + messages
        
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def __call__(self, message):
        # We invoke the graph with the new message
        # Persistence handles retrieving the old messages automatically
        input_data = {"messages": [{"role": "user", "content": message}]}
        output = self.app.invoke(input_data, config=self.config)
        
        # Return the content of the last assistant message
        return output["messages"][-1].content

# --- Usage ---
bot = Chatbot(system="You are a helpful assistant.")

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop your output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.


Your available actions are:
calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point
syntax if necessary

wikipedia:
e.g. wikipedia: Django
Returns a summary from searching Wikipedia

simon_blog_search:
e.g. simon_blog_search: Python
Search Simon's blog for that term

Example session:
Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: France
PAUSE

You will be called again with this:
Observation: France is a country. The capital is Paris.

You then output:
Answer: The capital of France is Paris

Please Note: if you get basic conversation questions like "hi","hello","how are you?",\n
you have to answer "hi","hello","i am good".
""".strip()

import httpx
def wikipedia(query):
    response = httpx.get("https://en.wikipedia.org/w/api.php", params={
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    })
    return response.json()["query"]["search"][0]["snippet"]

def simon_blog_search(query):
    response = httpx.get("https://datasette.simonwillison.net/simonwillisonblog.json", params={
        "sql": """
        select
          blog_entry.title || ': ' || substr(html_strip_tags(blog_entry.body), 0, 1000) as text,
          blog_entry.created
        from
          blog_entry join blog_entry_fts on blog_entry.rowid = blog_entry_fts.rowid
        where
          blog_entry_fts match escape_fts(:q)
        order by
          blog_entry_fts.rank
        limit
          1
        """.strip(),
        "_shape": "array",
        "q": query,
    })
    return response.json()[0]["text"]

def calculate(number):
    return eval(number)

print(calculate("2+2"))

known_actions = {
    "wikipedia": wikipedia,
    "calculate": calculate,
    "simon_blog_search": simon_blog_search
}

