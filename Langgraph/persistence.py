#When the graph exucation end to store the values we use persistence , it store the intermediate as well as final value
#Persistence is implemented using checkpointers
#during compiling of graph with a chechpointer , the checkpointers saves a checkpoint of the graph state at every super-step
#Superstep is the edges in between the nodes
#Those checkpoints are saved to a thread, which can be accessed after graph execution. Because threads allow access to graph’s state after execution,
# several powerful capabilities including human-in-the-loop, memory, time travel, and fault-tolerance are all possible.

#Thread is a unique ID , assigned to every  checkpoint by a checkpointer


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
print(graph.invoke({"foo": "", "bar":[]}, config))

# to get state histoy at each superstep
HIST = list(graph.get_state_history(config=config))
for h in HIST:
    print(h)
    print()
#To Replay from specific  superstep
"""config :RunnableConfig = {"configurable": {"thread_id":"1","checkpoint_id":"f100cca-3a16-6b80-8000-1c76ddd538eb"}}
   graph.invoke(None, config=config)"""

#If a particular thing need to be retains in mutiple thread we use Store
from langgraph.store.memory import InMemoryStore
in_memory_store = InMemoryStore()
user_id = "1"
namespace_for_memory = (user_id,"memories")