from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import get_answer

app = FastAPI(
    title="WSN Tutor API",
    description="AI-powered tutoring assistant for Wireless Sensor Networks",
    version="1.0.0"
)

# CORS - allows React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Request body model
class QueryRequest(BaseModel):
    question: str

# Response body model
class QueryResponse(BaseModel):
    answer: str

@app.get("/health")
def health_check():
    return {"status": "WSN Tutor API is running"}

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    answer = get_answer(request.question)
    return QueryResponse(answer=answer)