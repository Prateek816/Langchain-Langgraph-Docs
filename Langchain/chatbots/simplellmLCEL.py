import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
#print(groq_api_key)

from langchain_groq import ChatGroq

model = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

from langchain_core.messages import HumanMessage,SystemMessage
#Sytemmessage -> Instruction
#human - > inout prompt

messages = [
    SystemMessage(content="Translate the following from English to French"),
    HumanMessage(content = "Hello How are you?")

]
result = model.invoke(messages)
#print(result.content)
from langchain_core.output_parsers import StrOutputParser
parser =StrOutputParser()
ans = parser.invoke(result)
#print(ans)

#Using LCEL
chain = model|parser
chain.invoke(messages)


#Prompt Templates
from langchain_core.prompts import ChatPromptTemplate

generic_template = "Translate the fillowing into {language}:"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",generic_template),
        ("user","{text}")
    ]
)
prompt_invoked = (prompt.invoke({"language":"French","text":"Hello"}))

chain = prompt|model|parser
res =chain.invoke({"language":"French","text":"Hello , i have 10 underage childeren in my basement"})
print(res)


















