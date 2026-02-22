#chaining helps in pipeline the tasks , instead of manually connecting the components

#=================SIMPLE CHAINING=====================
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]
)
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

chain = prompt|model|parser

#print(chain.invoke({"topic":"AI"}))
#print(chain.get_graph().print_ascii())