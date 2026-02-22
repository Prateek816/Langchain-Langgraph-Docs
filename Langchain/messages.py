"""Messages are the fundamental unit of context for models in LangChain. They represent the input a
nd output of models, carrying both the content and metadata needed to represent the state of a conversation when interacting with an LLM."""
"""Message Types: ROLE+CONTENT+METADATA"""

#Basic Usage ->
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage
model = init_chat_model("gpt-5-nano")
system_msg = SystemMessage("You are a helpful assistant.")
human_msg = HumanMessage("Hello, how are you?")
# Use with chat models
messages = [system_msg, human_msg]
response = model.invoke(messages)  # Returns AIMessage

"""----------------------------------------------------------------------"""

#System Messages ->
# System messages set the behavior of the assistant ,it represent an initial set of instructions that primes the model’s behavior. 


#HUman Messages ->
#A HumanMessage represents user input and interactions. They can contain text, images, audio, files, and any other amount of multimodal content.
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)

#AI Messages ->
#An AIMessage represents the output of a model invocation. They can include multimodal data, tool calls, and provider-specific metadata that you can later access.

response = model.invoke("Explain AI")
print(type(response))  # <class 'langchain.messages.AIMessage'>

"""----------------------------------------------------------------------"""


