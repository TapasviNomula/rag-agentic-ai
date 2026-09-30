"""FastAPI interface. Run: uvicorn app:app --reload"""
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from src.graph import ask, build_rag_graph

app = FastAPI(title="Agentic AI RAG API")
graph = build_rag_graph()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: List[str]
    confidence_score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
def chat(request: QueryRequest):
    return ask(graph, request.query)
