import os
from dotenv import load_dotenv
load_dotenv()

def print_keys(data, indent=0):
    spacing = "    " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{spacing}🔹 {key}")
            print_keys(value, indent + 1)

    elif isinstance(data, list):
        for item in data:
            print_keys(item, indent)

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

response = tavily_tool.invoke({"query": "I want to Learn Python"})
for key , value in response.items():
    print(key)

response = tavily_tool.invoke({"query": "best Website to learn python"})
print(response['answer'])
response = tavily_tool.invoke({"query": "I want to Learn Python"})

text = response.get("results", "")

# limit to 150 words
words = text.split()[:1000]
final_text = " ".join(words)

print(final_text)