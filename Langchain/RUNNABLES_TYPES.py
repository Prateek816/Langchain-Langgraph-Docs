#Runnables are of two types -
#1. Task Specific Runnables 
#2. Runnables Primitives

#=========Runnable Sequence==========
#Runnable Sequence executes each step one after another, passing the ooutput tof one step as input to the next
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableSequence , RunnableParallel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile")
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
parser=StrOutputParser()
chain  = RunnableSequence(prompt ,model , parser)


#==========Runnable Parallel===============
#Runnable Parallel is a runnable primitive that allows multiple runnable to execute in parallel
#Each runnable receives the same input and processes deiffrently

prompt1=  PromptTemplate(
    template = "generate a tweet about {topic}",
    input_variables = ["topic"]
)
prompt2 = PromptTemplate(
    template = "generate a linkeding post about {topic}",
    input_variables = ["topic"]
)
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet' : RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2,model,parser)
})

print(parallel_chain.invoke({"topic":"AI"}))

#===============Runnable PassThrough============
#It just returns the input

from langchain_core.runnables import RunnablePassthrough
passthrough = RunnablePassthrough()
print(passthrough.invoke({"topic":"AI"}))


#============Runnable Lambda==============
#with help of runnable lambda we convert functions into runnables
from langchain_core.runnables import RunnableLambda

def word_counter(text):
    return len(text.split())

runnable_word_counter = RunnableLambda(word_counter)
prompt = PromptTemplate(
    template = "write a joke about {topic}",
    input_variables=['topic']
)
model = ChatGroq(model="llama-3.3-70b-versatile")
parser = StrOutputParser()
joke_gen_chain = RunnableSequence(prompt,model,parser)
parallel_chain = RunnableParallel({
    'joke':RunnablePassthrough(),
    'word_count':runnable_word_counter
})
final_chain = RunnableSequence(joke_gen_chain,parallel_chain)
result = final_chain.invoke({"topic":"AI"})
final_result = """{} \nword count - {}""".format(result['joke'],result['word_count'])
print(final_result)


#============Runnable Branch ===========

prompt_positive = PromptTemplate(
    template="Write an appropriate response to this positive feedback:\n{sentiment}",
    input_variables=["sentiment"]
)

prompt_negative = PromptTemplate(
    template="Write an appropriate response to this negative feedback:\n{sentiment}",
    input_variables=["sentiment"]
)


# ------------------------
# 5. Branch Router
# ------------------------
branch_chain = RunnableBranch(
    (lambda x: x['sentiment'].sentiment == 'positive', prompt_positive | model | parser_raw),
    (lambda x: x['sentiment'].sentiment == 'negative', prompt_negative | model | parser_raw),
    RunnableLambda(lambda x: "Could not determine sentiment.")
)


# ------------------------
# 6. Combined Chain
# ------------------------
full_chain = (
    RunnablePassthrough.assign(sentiment=classifier_chain)
    | branch_chain
)
