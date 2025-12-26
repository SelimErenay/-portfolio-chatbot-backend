from fastapi import FastAPI
from mangum import Mangum

app = FastAPI(title="Portfolio Chatbot API")

@app.post("/chat")
def chat(req: dict):
    return {
        "answer": "Backend is live.",
        "citations": []
    }

handler = Mangum(app)