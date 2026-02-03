#Managing the Conversation History
"""One important concept to understand when building chatbots is how to manage conversationd history. If left unmanaged , the list of messagess will grow unbounded and 
potentially overflow the context window of the LLM. Therefore,it is important to add a step that limits the size of the messages you are passing in"""


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found")

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key
)

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer all questions to the best of your ability in this {language}. "),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | model

# Session store
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Add message history
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="messages",
)

config = {
    "configurable": {
        "session_id": "chat3"
    }
}

# Invoke
response = with_message_history.invoke(
    {
        "messages": [HumanMessage(content="Hi! My name is Prateek")],"language":"French"
    },
    config=config
)
print(response.content)


from langchain_core.messages import trim_messages , AIMessage

trimmer = trim_messages(
    max_tokens=70,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human"
)

messages = [
    SystemMessage(content="You are a good Assistant"),
    HumanMessage(content="Hi! i am Pratee"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="What's 2+2"),
    AIMessage(content="4"),
    HumanMessage(content="thanls"),
    AIMessage(content="no problem")
]

print(trimmer.invoke(messages))

from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough

chain=(
    RunnablePassthrough.assign(messages=itemgetter("messages")|trimmer)
    |prompt
    |model
)

response = chain.invoke({
    "messages":messages + [HumanMessage(content="What Ice creame do i like")],
    "language":"English"
})

print(response.content)

#Lets wrap in Message history 
with_message_history  = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages"
)
config = {"configurable":{"session_id":"chat5"}}

response = chain.invoke({
    "messages":messages + [HumanMessage(content="What is MY name")],
    "language":"English"
})

print(response.content)
