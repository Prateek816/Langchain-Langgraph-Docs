from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.2,  # <-- Super slow! We can only make a request once every 5 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)
model = init_chat_model("openai/gpt-oss-120b",model_provider="groq",rate_limiter=rate_limiter)

while True:
    user_input = input("Enter your message (or 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    response = model.invoke([{"role": "user", "content": user_input}])
    print("Model response:", response)