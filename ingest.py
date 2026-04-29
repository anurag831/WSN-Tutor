from dotenv import load_dotenv
import os
import time
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# Step 1 - Load all PDFs from the documents folder
docs_path = "documents"
all_documents = []

for filename in os.listdir(docs_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(docs_path, filename))
        all_documents.extend(loader.load())

print(f"Loaded {len(all_documents)} pages from all PDFs")

# Step 2 - Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)
chunks = text_splitter.split_documents(all_documents)
chunks = [chunk for chunk in chunks if len(chunk.page_content.strip()) > 50]

print(f"Split into {len(chunks)} chunks")

# Step 3 - Load HuggingFace embedding model (downloads once, runs locally)
print("Loading HuggingFace embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully")

# Step 4 - Store in ChromaDB in batches
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_or_create_collection(name="wsn_tutor")

batch_size = 100

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    print(f"Embedding batch {i//batch_size + 1} of {-(-len(chunks)//batch_size)}...")

    texts = [chunk.page_content for chunk in batch]
    metadatas = [chunk.metadata for chunk in batch]
    ids = [f"chunk_{i+j}" for j in range(len(batch))]

    embedded_vectors = embeddings.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embedded_vectors,
        metadatas=metadatas,
        ids=ids
    )

print("Vectorstore created and saved successfully")
print(f"Total chunks stored: {collection.count()}")