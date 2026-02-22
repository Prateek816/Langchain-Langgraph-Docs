from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()

model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.1,
)