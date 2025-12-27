from fastapi import FastAPI
from pydantic import BaseModel

from rag.retriever import retrieve
from rag.llm import generate_answer

app = FastAPI(title="Portfolio Chatbot API")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    query: str
    answer: str


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # 1. Retrieve relevant chunks from FAISS
    chunks = retrieve(req.message, k=3)

    # 2. Generate final answer using OpenAI + retrieved context
    answer = generate_answer(
        question=req.message,
        context=chunks
    )

    # 3. Return final response
    return {
        "query": req.message,
        "answer": answer
    }
