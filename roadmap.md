## Steps

### 1. create a virtual environment (for dependency isolation and preventing conflicts)
`python -m venv venv`

### 2. activate the virtual environment
`venv\Scripts\activate`

### 3. Install dependencies
`pip install fastapi uvicorn langchain langchain-community langchain-google-genai chromadb pypdf python-dotenv`

### Dependency usage

#### The Dependencies Explained
fastapi
> The web framework that turns your Python functions into REST API endpoints. When your React frontend sends a question, FastAPI is what receives it and routes it to the right function. Think of it as Express.js but for Python.

uvicorn
> FastAPI can't run on its own — it needs a server. Uvicorn is that server. You'll always start your backend with uvicorn main:app the same way you'd use node index.js in Express.

langchain
> The core orchestration library. It provides the building blocks — document loaders, text splitters, chains, retrievers — that connect all the pieces of your RAG pipeline together. Without it you'd have to write all that glue code yourself.

langchain-community
> An extension of LangChain that contains community-contributed integrations — specifically the ChromaDB vector store integration and the PDF document loaders you'll use to read your 13 PDFs.

langchain-google-genai
> The official LangChain integration for Google's Gemini API. It lets you use Gemini as both the LLM (for generating answers) and the embedding model (for converting text to vectors) directly within LangChain's pipeline.

chromadb
> Your vector database. After your PDFs are chunked and converted to embeddings, ChromaDB stores them locally on disk. When a user asks a question, ChromaDB finds the most relevant chunks using similarity search.

pypdf
> A Python library that reads PDF files and extracts their text content. LangChain's PDF loader uses pypdf under the hood to open your 13 PDFs and pull out the raw text before chunking.

python-dotenv
> Reads your .env file and loads the Gemini API key into your environment variables. This keeps your API key out of your code so you don't accidentally push it to GitHub.

```
pypdf          → reads PDFs
langchain      → chunks the text, orchestrates the pipeline
langchain-google-genai → converts chunks to embeddings via Gemini
chromadb       → stores and searches those embeddings
langchain-community → connects LangChain to ChromaDB
fastapi+uvicorn → exposes the pipeline as a REST API
python-dotenv  → keeps your API key safe
```

### 4. Getting LLM API key
- created a new project in google ai studio and added api keys in it (free tier)

### 5. Test the LLM api working

- created a simple file called test_gemini.py with the following content:

```
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key
)

response = llm.invoke("What is a Wireless Sensor Network in one sentence?")
print(response.content)
```

- this just calls the gemini-2.5-flash model and gives our prompt which returns a response in the terminal

### 6. creating the vector databse