# Dockerfile — Agent RAG Santé
# Compatible Render (Web Service, Docker) et Hugging Face Spaces (SDK: Docker)

FROM python:3.11-slim

WORKDIR /app

# Dépendances système minimales pour sentence-transformers / chromadb
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer les dépendances Python d'abord (cache Docker plus efficace)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste du projet
COPY . .

# Construire la base vectorielle au moment du build, pas au runtime
# (ainsi vectordb/ est prêt dès le premier démarrage du conteneur)
RUN python ingest.py

# Render fournit $PORT dynamiquement ; Hugging Face Spaces attend le port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
