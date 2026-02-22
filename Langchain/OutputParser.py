#LLMs which are not tuned to worked with structuredouput in langchain , can still be used to give structural output with the help of outputparser
#Output parser helps convert LLM reponses into structured format like json , csv , pydantic models
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , JsonOutputParser 
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema , PydanticOutputParser

load_dotenv()



#StrOutput parser is the simplest out parser in langchain , it returns a a plain string

#JSON OutputParser - 

model = ChatGroq(model="llama-3.3-70b-versatile")
parser = JsonOutputParser()
template = PromptTemplate(
    template="Give me the name , age and city of a fictional person \n {format_instructions} ",
    partial_variables={'format_instructions':parser.get_format_instructions()}
)

prompt = template.format()
result = model.invoke(prompt)
print(result)
print(parser.parse(result.content))
print(prompt)

#or simply create chain
chain = template | model |parser
print(chain.invoke({}))


#StructuredOutputparser

schema = [
    ResponseSchema(name='fact_1',description = 'fact 1 about the topic'),
    ResponseSchema(name='fact_2',description = 'fact 2 about the topic'),
    ResponseSchema(name='fact_3',description = 'fact 3 about the topic'),

]
parser=StructuredOutputParser.from_response_schemas(schema)
template = PromptTemplate(
    template="Give 3 facts about \n{topic} \n {format_instruction}",
    partial_variables={'format_instruction':parser.get_format_instructions()},
    input_variables=['topic']
)
chain = template|model|parser
print(chain.invoke({"topic":"AI"}))

from pydantic import BaseModel , Field

class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age: int = Field(gt=18 , description="Age of the person")
    city: str = Field(description="Name of the city the person belogs to")

parser = PydanticOutputParser(pydantic_object=Person)
template = PromptTemplate(
    template="Give me the name , age and city of a fictional {place} \n {format_instructions}",
    partial_variables={'format_instructions':parser.get_format_instructions()},
    input_variables=['place']
)

chain = template | model | parser
print(chain.invoke({"place":"india"}))
