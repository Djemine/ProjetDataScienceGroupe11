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
import os
import hashlib
import glob
from rag_engine import RAGEngine



def _compute_data_hash():
    txt_files = sorted(glob.glob("data/*.txt"))
    hasher = hashlib.sha256()
    for filepath in txt_files:
        with open(filepath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def _needs_ingestion():
    if not os.path.isdir("vectordb") or not os.listdir("vectordb"):
        return True

    hash_file = "vectordb/.data_hash"
    if not os.path.exists(hash_file):
        return True

    with open(hash_file) as f:
        stored_hash = f.read().strip()

    return stored_hash != _compute_data_hash()

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
    import subprocess

    if _needs_ingestion():
        print(" Base vide ou data/ modifié — ingestion en cours...")
        from ingest import run_ingestion
        run_ingestion()
    else:
        print(" Base vectorielle déjà à jour, ingestion ignorée.")

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