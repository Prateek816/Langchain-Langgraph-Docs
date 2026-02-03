from pydantic import BaseModel, Field
class MovieInfo(BaseModel):
    title: str = Field(..., description="The title of the movie")
    director: str = Field(..., description="The director of the movie")
    release_year: int = Field(..., description="The year the movie was released")
    summary: str = Field(..., description="A brief summary of the movie plot")
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
load_dotenv()
model = init_chat_model(
    model="llama-3.1-8b-instant",
    model_provider="groq",
    temperature=0.1,
)
model_with_structure = model.with_structured_output(MovieInfo)
response = model_with_structure.invoke("Provide information about the movie Inception.")
print(response)  