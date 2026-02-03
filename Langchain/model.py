#LLMs are powerful AI tools that can interpret and generate text like humans. They’re versatile enough to write content, translate languages, summarize, 
# and answer questions without needing specialized training for each task.

#In addition to text generation, many models support:
 #Tool calling - calling external tools (like databases queries or API calls) and use results in their responses.
 #Structured output - where the model’s response is constrained to follow a defined format.
 #Multimodality - process and return data other than text, such as images, audio, and video.
 #Reasoning - models perform multi-step reasoning to arrive at a conclusion.

# The quality and capabilities of the model you choose directly impact your agent’s baseline reliability and performance.(important) 
# Different models excel at different tasks - some are better at following complex instructions, others at structured reasoning, 
# and some support larger context windows for handling more information.
import os
from dotenv import load_dotenv
load_dotenv() 
from langchain_google_genai import GoogleGenerativeAI
#A way to initialize a model directly using the provider package:
llm = GoogleGenerativeAI( # see other parameters too for model configuration
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=1024 #tokens usually are words or word pieces (subwords)
)

"---------------------------------------------------------------"


#Using INIT CHAT MODEL ->
from langchain.chat_models import init_chat_model
model = init_chat_model(
    model="gemini-2.5-flash-lite", 
    model_provider="google_genai",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
    max_tokens=1024
)
response = llm.generate(["What is the capital of France?"])
print(response.generations[0][0].text)  # Output: Paris
response = llm.invoke("Tell me a joke.")
print(response)  # Output: A funny joke

"---------------------------------------------------------------"


#A list of messages can be provided to a chat model to represent conversation history. Each message has a role that models use to indicate who sent the message in the conversation.
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who won the world series in 2020?"),
    AIMessage(content="The Los Angeles Dodgers won the World Series in 2020."),
    HumanMessage(content="Where was it played?")
]
response  = llm.invoke(messages)
print(response)  

"---------------------------------------------------------------"

#For Streaming ->
"""Calling stream() returns an iterator that yields output chunks as they are produced. You can use a loop to process each chunk in real-time:"""

response = model.stream("Tell me a story about a brave knight.")
for chunk in response:
    print(chunk.content, end='', flush=True)  # Print each chunk as it arrives
    print()  # For a new line after the story is complete

"---------------------------------------------------------------"

#Batch Processing ->
#You can send multiple prompts in a single request using the batch() method. This is more efficient than sending individual requests for each prompt.
responses = model.batch([
    "Explain the theory of relativity.",
    "Write a short poem about the ocean.",
    "What are the benefits of meditation?"
])

for r in responses:
    print(r.content)

#By default, batch() will only return the final output for the entire batch. If you want to receive the output for each individual input as it finishes generating, you can stream results with batch_as_completed():

for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)

"---------------------------------------------------------------"

#Structured Output ->
#We can define a schema for the model's output using Pydantic models. The model will generate responses that conform to the specified structure.

from pydantic import BaseModel, Field
class MovieInfo(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
model_with_structure = model.with_structured_output(MovieInfo)
response = model_with_structure.invoke("Provide information about the movie Inception.")
print(response)  

