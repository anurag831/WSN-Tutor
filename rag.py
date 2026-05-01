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

# Prompt template
prompt_template = PromptTemplate(
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

def get_answer(query: str) -> str:
    # Step 1 - Embed the query
    query_embedding = embeddings.embed_query(query)

    # Step 2 - Search ChromaDB for top 5 similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    # Step 3 - Extract the retrieved chunks
    retrieved_chunks = results["documents"][0]
    context = "\n\n".join(retrieved_chunks)

    # Step 4 - Build the prompt
    prompt = prompt_template.format(
        context=context,
        question=query
    )

    # Step 5 - Send to Gemini and get answer
    response = llm.invoke(prompt)
    return response.content

if __name__ == "__main__":
    test_query = "How to control congestion in wsn using active queue management"
    answer = get_answer(test_query)
    print(answer)