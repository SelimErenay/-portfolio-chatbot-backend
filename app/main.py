from fastapi import FastAPI
from pydantic import BaseModel

from rag.retriever import retrieve

app = FastAPI(title="Portfolio Chatbot API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    query: str
    context: list[str]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    chunks = retrieve(req.message, k=3)

    return {
        "query": req.message,
        "context": chunks
    }