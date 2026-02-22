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


