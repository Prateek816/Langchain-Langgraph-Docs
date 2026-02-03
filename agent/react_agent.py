import os
import requests
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Load your .env file
load_dotenv()

# 2. Initialize the Groq Model
# Temperature 0 is generally better for ReAct agents to stay on track
model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0, 
)

# 3. Define Tools
search_tool = DuckDuckGoSearchRun()

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city using the WeatherStack API.
    """
    api_key = "537760f9f17b59f55c4d45114532a1dd"
    url = f'https://api.weatherstack.com/current?access_key={api_key}&query={city}'
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check if the API returned an error (like invalid key)
        if "current" in data:
            weather = data["current"]
            return f"Weather in {city}: {weather['weather_descriptions'][0]}, Temperature: {weather['temperature']}°C"
        else:
            return f"Could not find weather for {city}. Error: {data.get('error', {}).get('info', 'Unknown error')}"
    except Exception as e:
        return f"Error connecting to weather service: {str(e)}"

tools = [search_tool, get_weather_data]

# 4. Create the Agent using LangGraph (the modern replacement for AgentExecutor) , for multiple nature of agent use AgentExecutor
agent_executor = create_react_agent(model, tools)

# 5. Invoke the Agent
query = "Find the capital of France then find its current weather condition"

print("--- Agent is thinking ---")
try:
    response = agent_executor.invoke({"messages": [("user", query)]})
    
    # Get the last message from the result
    final_message = response["messages"][-1]
    print("\n--- FINAL ANSWER ---")
    print(final_message.content)
except Exception as e:
    print(f"\nExecution failed: {e}")


#Code for AgentExecutor - 

from langchain_classic.agents import AgentType
from langchain.agents import load_tools 
from langchain_classic.agents import initialize_agent
tool=load_tools(["wikipedia"],llm=llm)
agent=initialize_agent(tool,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True) #note how to agent type
agent.run("what is llama and who create this llm model?")

from langchain_classic import hub
prompt=hub.pull("hwchase17/openai-functions-agent")
from langchain.agents import create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "hello how are you?"})






