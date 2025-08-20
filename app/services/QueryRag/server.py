from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import rag_answer

app = FastAPI(title="RAG+LLM API")

class AskReq(BaseModel):
    query: str

@app.post("/ask")
async def ask(req: AskReq):
    return await rag_answer(req.query)

@app.get("/")
async def root():
    return {"ok": True}
