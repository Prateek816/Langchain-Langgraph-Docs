"""Agents combine language models with tools to create systems that can reason about tasks, decide which tools to use, and iteratively work towards solutions."""

from langchain.agents import create_agent

#Static models are configured once when creating the agent and remain unchanged throughout execution.

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    api_key = os.getenv("GOOGLE_API_KEY"),
    temperature = 0
)
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful research assistant."
)

"---------------------------------------------------------------"

#For more control over the model configuration, initialize a model instance directly using the provider package. 
from langchain_ollama import OllamaLLM
model = OllamaLLM(
    model="gemma:2b",
    temperature=0.1,
    top_k= 1000 ,#or max_token , control how many text is generated
    
)

#now create agent
agent = create_agent(model=model,tools=tool)

#in above code langchain just tell model , here are tools now tell me which tool is best
#above code will give error because OllamaLLM, GROQ model doesnt support built-in native tool-calling
"---------------------------------------------------------------"

# to use Ollama model as tool router(agent) ->
"""couldnt resolve create_react_agent error"""
from langchain.agents import create_agent
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model = "gemma:2b")
tools = []
prompt = PromptTemplate.from_template(
    """You are an agent that can use tools.

Tools:
{tools}

Question: {input}

{agent_scratchpad}
"""
)
from langgraph.prebuilt import create_react_agent
agent = create_react_agent(llm,tools,prompt)
#In LangChain, create_react_agent is a high-level constructor used to build an agent that follows the ReAct logic (Reasoning + Acting). 
# It allows an LLM to "think" about a problem, choose a tool, observe the output, and repeat until it finds an answer.

"---------------------------------------------------------------"

#Dynamic models are selected at runtime based on the current state and context. This enables sophisticated routing logic and cost optimization.

from langchain_google_genai import GoogleGenerativeAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call , ModelRequest , ModelResponse

basic_model = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
advanced_model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

@wrap_model_call
def dynamic_model_selection(request:ModelRequest , handler)->ModelResponse:
    """Chooes Model based on conversation complexity"""
    message_count = len(request.state["messages"])
    if message_count >10:
        model = advanced_model
    else :
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model = basic_model ,# DEFAULT
    tools = tools,
    middleware=[dynamic_model_selection]
)

"---------------------------------------------------------------"

"""SYSTEM PROMPT"""
#You can shape how your agent approaches tasks by providing a prompt. The system_prompt parameter can be provided as a string:
#The system_prompt parameter accepts either a str or a SystemMessage. Using a SystemMessage gives you more control over the prompt structure

#DYNAMIC SYSTEM PROMPT

from typing import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest


class Context(TypedDict):
    user_role: str

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.get("user_role", "user")
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt

agent = create_agent(
    model="gpt-4o",
    tools=[web_search],
    middleware=[user_role_prompt],
    context_schema=Context
)

# The system prompt will be set dynamically based on context
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    context={"user_role": "expert"}
)

"---------------------------------------------------------------"


