from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
model = os.getenv("GEMINI_MODEL")

llm = ChatGoogleGenerativeAI(
    model=model,
    google_api_key=api_key
)

response = llm.invoke("What is a Wireless Sensor Network in one sentence?")
print(response.content)