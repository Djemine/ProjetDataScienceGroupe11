"""
ingest.py
Étape 1 du RAG : lit les fichiers texte du dossier data/, les découpe en chunks,
les vectorise avec sentence-transformers, et les stocke dans ChromaDB.

Usage : python ingest.py
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
import hashlib

DATA_DIR = "data"
DB_DIR = "vectordb"
COLLECTION_NAME = "sante_docs"

# Modèle d'embeddings open source, gratuit, tourne en local (pas d'API key nécessaire)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800       # caractères par chunk
CHUNK_OVERLAP = 150    # chevauchement entre chunks pour ne pas couper le contexte


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Découpe un texte en chunks avec chevauchement, en essayant de couper sur des sauts de paragraphe."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # Si le paragraphe seul dépasse déjà chunk_size, on le découpe brutalement
            if len(para) > chunk_size:
                for i in range(0, len(para), chunk_size - overlap):
                    chunks.append(para[i:i + chunk_size])
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks

def compute_data_hash():
    """Empreinte du contenu actuel de data/, pour savoir si une ré-ingestion est nécessaire."""
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    hasher = hashlib.sha256()
    for filepath in txt_files:
        with open(filepath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def main():
    print(" Démarrage de l'ingestion...")

    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Repartir d'une collection propre à chaque exécution du script
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))
    if not txt_files:
        print(f" Aucun fichier .txt trouvé dans {DATA_DIR}/")
        return

    all_ids, all_docs, all_metadatas = [], [], []
    chunk_counter = 0

    for filepath in txt_files:
        source_name = os.path.basename(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"  - {source_name} : {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            chunk_counter += 1
            all_ids.append(f"chunk_{chunk_counter}")
            all_docs.append(chunk)
            all_metadatas.append({"source": source_name, "chunk_index": i})

    collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)

    print(f"\n Ingestion terminée : {chunk_counter} chunks stockés dans la collection '{COLLECTION_NAME}'.")
    print(f" Base vectorielle sauvegardée dans le dossier '{DB_DIR}/'.")
    with open(os.path.join(DB_DIR, ".data_hash"), "w") as f:
        f.write(compute_data_hash())

def run_ingestion():
    main()


if __name__ == "__main__":
    main()
