"""
main.py
API FastAPI qui expose l'agent RAG et sert l'interface web statique.

Lancement : uvicorn main:app --reload
Puis ouvrir : http://127.0.0.1:8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

from rag_engine import RAGEngine

app = FastAPI(title="Agent d'Orientation Médicale & Prévention - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine: Optional[RAGEngine] = None


@app.on_event("startup")
def load_engine():
    global rag_engine
    rag_engine = RAGEngine()
    print(" RAGEngine chargé et prêt.")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list
    analysis: Optional[dict] = None


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="Le moteur RAG n'est pas encore prêt.")

    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]

    result = rag_engine.generate_answer(request.message, history=history)
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        analysis=result.get("analysis"),
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "engine_loaded": rag_engine is not None}


# Sert l'interface web (dossier static/) à la racine
app.mount("/", StaticFiles(directory="static", html=True), name="static")