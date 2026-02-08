#===========CONDITIONAL CHAINING===========

from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough


# ------------------------
# 1. Pydantic Model
# ------------------------
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(
        description="Give the sentiment of the feedback, either 'positive' or 'negative'"
    )


# ------------------------
# 2. Parsers and Model
# ------------------------
parser_raw = StrOutputParser()
parser_struct = PydanticOutputParser(pydantic_object=Feedback)

model = ChatGroq(model="llama-3.3-70b-versatile")


# ------------------------
# 3. Classifier Prompt
# ------------------------
prompt_classifier = PromptTemplate(
    template=(
        "Classify the sentiment of the following feedback text into positive or negative:\n"
        "{feedback}\n\n"
        "{format_instructions}"
    ),
    input_variables=["feedback"],
    partial_variables={'format_instructions': parser_struct.get_format_instructions()}
)

classifier_chain = prompt_classifier | model | parser_struct


# ------------------------
# TEST classifier alone
# ------------------------
print("Classifier Output:", classifier_chain.invoke(
    {"feedback": "This is Terrible Smartphone"}
).sentiment)


# ------------------------
# 4. Branch Prompts
# ------------------------
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


# ------------------------
# 7. Run full system
# ------------------------
output = full_chain.invoke({"feedback": "This is Terrible Smartphone"})
print("\nFinal Routed Response:", output)


# ------------------------
# 8. Print graph
# ------------------------
print("\nChain Graph:")
print(full_chain.get_graph().print_ascii())
