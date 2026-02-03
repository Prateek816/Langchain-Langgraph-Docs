import os
from dotenv import load_dotenv
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables.")

from langchain_tavily import TavilySearch

tavily_tool = TavilySearch(
    api_key=TAVILY_API_KEY,
    max_results=5,
    topic="general",
    include_answer=True,          # <-- REQUIRED
    include_images=False,
    include_raw_content=True,
)

response = tavily_tool.invoke({"query": "Recent news of India"})
print(response)
