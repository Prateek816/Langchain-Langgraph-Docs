#to check wheter the user has entered correct data type or not
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

class State(BaseModel):
    name:str

def experiment(state:State):
    return {"name":state.name}

builder = StateGraph(State)
builder.add_node("experiment",experiment)
builder.add_edge(START,"experiment")
builder.add_edge("experiment",END)

graph = builder.compile()
print(graph.invoke(State(name="hi")))
