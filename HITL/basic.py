from typing import Annotated , TypedDict , List
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph , START , END
from langgraph.types import Command , interrupt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage , BaseMessage
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama-3.3-70b-versatile")

class ChatState(TypedDict):
    messages: List[BaseMessage]

def chat_node(state:ChatState)->ChatState:
    decision =interrupt({
        "type":"approval",
        "reason":"Model is about to answer a user question",
        "question": state['messages'][-1].content,
        "instruction":"Approve this question ? yes/no",
    })
    if decision["approved"]=='no':
        return {"messages":[AIMessage(content="Your question was not approved by the human in the loop. Please modify your question and try again.", additional_kwargs={}, response_metadata={}, id="")]}
    
    else:
        response = llm.invoke(state['messages'])
        return {"messages": state['messages'] + [response]}
    
builder = StateGraph(ChatState)

builder.add_node("chat_node",chat_node)
builder.add_edge(START,"chat_node")
builder.add_edge("chat_node",END)
checkpoint_saver = InMemorySaver()
app = builder.compile(checkpointer =checkpoint_saver)

config = {"configurable":{"thread_id":"thread1"}}

initial_input = {
    "messages":[HumanMessage(content="What is the capital of France?", additional_kwargs={}, response_metadata={}, id="")]
}
result = app.invoke(initial_input,config=config)
print(result)

user_response = input("Do you approve the model's response? (yes/no): ")

final_result = app.invoke(
    Command(resume={"approved": user_response}),
    config=config
)
print(final_result)