#Short Term memory(thread-level persistence) enables agents to track multi turn conversations.

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

checkpointer = InMemorySaver()
builder =  StateGraph(...)
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages":[{"role":"user","content":"hello"}]},
    {"configurable":{"thread_id":"1"}
)

