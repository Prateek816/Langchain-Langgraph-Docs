#implement safety checks and content filtering for agents

"""Common use cases include:
Preventing PII leakage
Detecting and blocking prompt injection attacks
Blocking inappropriate or harmful content
Enforcing business rules and compliance requirements
Validating output quality and accuracy"""

from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="llama-3.3-70b-versatile")
agent = create_agent(
    model=llm,
    middleware=[
        PIIMiddleware(
            "email",
            strategy="mask",
            apply_to_input=True,
        ),
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        PIIMiddleware(
            pii_type="api_key",
            detector=r"sk-[A-Za-z0-9]{32}",   # correct custom regex
            strategy="block",                # block if matched
            apply_to_input=True,
        ),
    ],
)
result = agent.invoke({
    "messages": [{"role": "user", "content": "My email is john.doe@example.com and card is 5105-1051-0510-5100"}]
})
#print(result['messages'])
"""[HumanMessage(content='My email is john.doe@****.com and card is ****-****-****-5100', additional_kwargs={}, response_metadata={}, id='175c6b8c-b482-4b45-ad86-71341a533413'), AIMessage(content="I see you've provided some sensitive information. For security purposes, I want to remind you that it's not recommended to share your email address or credit card details publicly. \n\nIf you're trying to get help with something related to your email or credit card, I'd be happy to assist you in a more general way. Could you please tell me what's on your mind, and I'll do my best to provide guidance without requiring any sensitive information?"""
