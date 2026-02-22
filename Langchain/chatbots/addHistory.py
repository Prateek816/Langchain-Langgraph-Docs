import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)

from langchain_core.messages import SystemMessage,HumanMessage

temp =model.invoke([HumanMessage(content="Hi , My name is Prateek , I am a Btech Student")])
#print(temp.content)

from langchain_core.messages import AIMessage
temp = model.invoke([
    HumanMessage(content="Hi , My name is Prateek , I am a Btech Student"),
    AIMessage(content="Hello Prateek! 👋 Nice to meet you. How can I help you today? Whether it’s coursework, projects, career advice, or just a chat about tech, I’m here for you. 🚀"),
    HumanMessage(content="Hey What's my name")
])
#print(temp.content)


##Message History
#We can use a Message History class to wrap our model and make it statefu;. This will keep track of
#inputs and outputs of the model, and store them in some datastore. Future interations will then load
#thos messages and pass them into the chain as part of the input. Let's see how to use this


from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store={}
def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]



with_message_history  = RunnableWithMessageHistory(model,get_session_history) #infuse model with history
config = {"configurable":{"session_id":"chat1"}}

res = with_message_history.invoke([
    HumanMessage(content="Hello , my name is Prateek Rastogi")],
    config = config
)
print(store["chat1"])
res = with_message_history.invoke([
    HumanMessage(content="What is my name")],
    config = config
)
print(res.content)

#channge the config -> change the session id
config1 = {"configurable":{"session_id":"chat2"}}
res = with_message_history.invoke(
    [HumanMessage(content="What is my name")], # dont remembers the history now
    config = config1
)
print(res.content)



