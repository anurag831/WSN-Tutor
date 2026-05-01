from dotenv import load_dotenv
import os
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL")

# Load the same embedding model used during ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Connect to the existing vectorstore
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_collection(name="wsn_tutor")

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model=gemini_model,
    google_api_key=api_key
)

# Project document source names — add all your project doc names here
PROJECT_DOC_SOURCES = [
    "documents\\projectDoc1.pdf",
    "documents\\projectDoc2.pdf",
    "documents\\projectDoc3.pdf",
    "documents\\projectDoc4.pdf",
    "documents\\projectDoc5.pdf",
    "documents\\projectDoc6.pdf",
    "documents\\projectDoc7.pdf",
    "documents\\projectDoc8.pdf",
    "documents\\projectDoc9.pdf"
]

# Keywords that signal the user is asking about your specific project
PROJECT_KEYWORDS = [
    "our project", "our solution", "our approach", "our work",
    "our congestion control", "our algorithm", "our proposed",
    "our method", "our system", "our implementation"
]

# Prompt template for general WSN questions
general_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are WSN Tutor, a helpful and friendly academic assistant that explains
Wireless Sensor Networks and congestion control concepts to students.

Use the following context retrieved from academic documents to answer the
student's question. Explain clearly and simply as if teaching a student.
If the answer is not in the context, say so honestly and provide
a general explanation if you can.

Context:
{context}

Student's Question:
{question}

Answer:
"""
)

# Prompt template for project-specific questions
project_prompt_template = PromptTemplate(
    input_variables=["project_context", "general_context", "question"],
    template="""
You are WSN Tutor, a helpful and friendly academic assistant.

The student is asking about a specific congestion control project developed 
for Wireless Sensor Networks. The project uses Active Queue Management based 
on Fuzzy Logic to control congestion in WSNs.

Use the PROJECT DOCUMENTATION below as the primary source to answer the question.
Use the GENERAL CONTEXT only to supplement if needed.

PROJECT DOCUMENTATION:
{project_context}

GENERAL CONTEXT:
{general_context}

Student's Question:
{question}

Answer:
"""
)

def is_project_query(query: str) -> bool:
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in PROJECT_KEYWORDS)

def get_answer(query: str) -> str:
    # Embed the query
    query_embedding = embeddings.embed_query(query)

    if is_project_query(query):
        # Retrieve chunks specifically from project documents
        project_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            where={"source": {"$in": PROJECT_DOC_SOURCES}}
        )

        # Retrieve general WSN chunks for supplementary context
        general_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            where={"source": {"$nin": PROJECT_DOC_SOURCES}}
        )

        project_context = "\n\n".join(project_results["documents"][0])
        general_context = "\n\n".join(general_results["documents"][0])

        prompt = project_prompt_template.format(
            project_context=project_context,
            general_context=general_context,
            question=query
        )

    else:
        # Normal retrieval for general WSN questions
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=5
        )
        context = "\n\n".join(results["documents"][0])
        prompt = general_prompt_template.format(
            context=context,
            question=query
        )

    response = llm.invoke(prompt)
    return response.content
