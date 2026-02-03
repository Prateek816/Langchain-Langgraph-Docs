from typing_extensions import TypedDict
from typing import Literal

class TypedDictState(TypedDict):
    name:str
    game:Literal["cricket","badmintion"]

def play_game(state:TypedDictState):
    print("play_game node has been called")
    return {"name":state['name'] + " want to play "}

def badmintion(state:TypedDictState):
    print("badmintion_node has been called")
    return {"name":state["name"] + " badminton","game":"badminton"}

def cricket(state:TypedDictState):
    print("cricket node has been called")
    return {"name":state["name"]+" cricket","game":"cricket"}

import random
def decide_play(state:TypedDictState)->Literal["cricket","badmintion"]:
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:
        return "cricket"
    else:
        return "badmintion"
    
from langgraph.graph import StateGraph, START, END
builder = StateGraph(TypedDictState)
builder.add_node("play_game",play_game)
builder.add_node("cricket",cricket)
builder.add_node("badmintion",badmintion)

builder.add_edge(START,"play_game")
builder.add_conditional_edges("play_game",decide_play)
builder.add_edge("cricket",END)
builder.add_edge("badmintion",END)
graph = builder.compile()

print(graph.invoke({'name':'badmintion'}))