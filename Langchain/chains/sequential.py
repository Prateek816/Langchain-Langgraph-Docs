
#=============SEQUENTIAL CHAINING=========================
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

parser = StrOutputParser()
prompt1 = PromptTemplate(
    template = "Generate a detailed report on {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template="generate a 5 pointer summar from a following text \n {text}",
    input_variables = ["text"]
)
model = ChatGroq(model="llama-3.3-70b-versatile")

chain = prompt1|model|parser|prompt2|model|parser

response =  chain.invoke({"topic":"unemployement in india"})
print(response)
print(chain.get_graph().print_ascii())

