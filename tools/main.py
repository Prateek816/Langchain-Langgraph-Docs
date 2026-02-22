#There are three ways to make tooles in LangChain:
#1. Using the @tool decorator
#2. Subclassing the Tool class
#3. Using the Tool.from_function method

from langchain.tools import tool , BaseTool

@tool("adding") 
def add_numbers(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b
response = add_numbers.invoke({"a": 5, "b": 10})
print(f"Result of adding numbers: {response}")
print(add_numbers.description)
print(add_numbers.name)
print(add_numbers.args)

#A Structured Tool in Lnagchain is a special type of tool that allows for structured input and output.
#Structured Tools are defined using Pydantic models to specify the schema for the input and output
#from langchain.tools import StructuredTools
from pydantic import BaseModel, Field
class MultiplyInput(BaseModel):
    a: int = Field(..., description="The first number to multiply")
    b: int = Field(..., description="The second number to multiply")
def multiply_func(a:int , b:int)->int:
    return a*b
"""multiply_tool = StructuredTool.from_function(
    func=multiply_func,
    name="multiply",
    description="Multiply two numbers",
    args_schema = MultiplyInput
)
"""

#using BaseTool Class

from langchain.tools import BaseTool
class MultiplyTool(BaseTool):
    name:str = "multiply"
    description:str = "Multiply Two numbers"
    def _run (self , a:int,b:int):
        return a*b
multiply_tool = MultiplyTool()
response = multiply_tool.invoke({"a": 5, "b": 10})
print(f"Result of multiplying numbers: {response}")
print(multiply_tool.description)
print(multiply_tool.name)
print(multiply_tool.model_json_schema)

"""----------------------------------------------------------------------"""

"""Tool Binding"""
#Tool Binding is the step where you register tools with a Language Model(LLM) :
#1.The LLM knows what tools are available
#2.The LLM can decide when to use a tool based on the user input
#3.The LLM can format the input and output for the tools correctly

"""---------------------------------"""

#tool calling
import os
from dotenv import load_dotenv
load_dotenv()
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.1,
)
@tool("Multiply")
def multiply(a:int , b:int)->int:
    "Given two number a and b this tools returns their product"
    return a*b
response = model.invoke("What is the product of 6 and 7?")
print(response.content)

@tool
def prateek_info()->str:
    "Who is Prateek Rastogi"
    return """Prateek Rastogi is a software engineer with 5 years of experience in full-stack development. He specializes in building scalable web applications and has a strong background in Python and JavaScript."""
model = model.bind_tools([multiply, prateek_info])
response = model.invoke("Who is Prateek Rastogi")
print(response.tool_calls[0])

"""---------------------------------"""

#Tool Execution
#Tool Execution is the process where the LLM decides to use a tool based on the user input
# now its work of user or langchain to run the tool
tool_to_execute = response.tool_calls[0]
result = prateek_info.invoke(tool_to_execute)
print(f"Tool Execution Result: {result}") #the output is well wrapped (ToolMessage) which contains metadata about the tool execution along with the actual result.
#Now we should print the answer in well mannered way
from langchain_core.messages import AIMessage , ToolMessage,HumanMessage,SystemMessage
query = "Who is Prateek Rastogi"
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=query),
    response,
    result
]
print(messages)
