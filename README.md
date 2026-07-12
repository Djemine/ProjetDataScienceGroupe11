# Assistant d'Orientation Médicale & Prévention (Option 3)

Agent IA RAG (Retrieval-Augmented Generation) qui répond à des questions de santé de premier niveau (paludisme, dengue, nutrition, pharmacies de garde) à partir d'une base de connaissances locale.

## Architecture

```
Question utilisateur
      │
      ▼
Recherche vectorielle (ChromaDB + embeddings sentence-transformers)
      │
      ▼
Contexte pertinent récupéré (chunks + sources)
      │
      ▼
LLM (Groq - Llama 3.3 70B) avec prompt système anti-hallucination
      │
      ▼
Réponse + sources citées
```

## Installation (5 minutes)

1. **Créer un environnement virtuel et installer les dépendances**
```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

2. **Obtenir une clé API Groq gratuite**
   - Aller sur https://console.groq.com
   - Créer un compte gratuit
   - Générer une clé API (section "API Keys")

3. **Configurer la clé**
```bash
cp .env.example .env
# puis éditer .env et coller ta clé : GROQ_API_KEY=gsk_xxxxx
```

## Utilisation

### Étape 1 — Ingestion (vectorisation des documents)
```bash
python ingest.py
```
Cela lit tous les fichiers `.txt` du dossier `data/`, les découpe en chunks, les transforme en vecteurs et les stocke dans `vectordb/` (ChromaDB).

### Étape 2 — Lancer l'application
```bash
uvicorn main:app --reload
```
Puis ouvrir : http://127.0.0.1:8000

### Étape 3 — Évaluer le système (rapport technique)
```bash
python evaluate.py
```
Ce script teste :
- **La pertinence** : la recherche vectorielle remonte-t-elle le bon document source pour chaque question ?
- **Le taux d'hallucination** : l'agent dit-il "je ne sais pas" pour des questions hors de sa base de connaissances (au lieu d'inventer une réponse) ?

Les résultats affichés (scores en %) sont directement réutilisables dans le rapport technique (section Évaluation).

## Structure du projet

```
agent-sante/
├── data/                    # Base de connaissances (à enrichir avec de vraies données)
│   ├── paludisme.txt
│   ├── dengue.txt
│   ├── nutrition.txt
│   └── pharmacies_garde.txt  # ⚠️ données d'exemple, à remplacer par de vraies données scrapées
├── vectordb/                # Base vectorielle ChromaDB (générée par ingest.py)
├── static/
│   └── index.html           # Interface web du chat
├── ingest.py                 # Étape 1 : collecte + vectorisation
├── rag_engine.py              # Étape 2 : logique de l'agent (recherche + LLM)
├── main.py                   # Étape 3 : API FastAPI
├── evaluate.py                # Étape 4 : évaluation du système
├── requirements.txt
└── .env.example
```

## Ce qu'il te reste à faire pour finaliser le projet

1. **Remplir `data/pharmacies_garde.txt` avec de vraies données.**
   Le fichier actuel contient des données d'exemple clairement indiquées comme telles.
   Idée simple si tu manques de temps : contacte/vérifie manuellement 5 à 10 pharmacies de ta ville et note leurs infos de garde réelles — ça suffit pour un prototype de démonstration. Si tu veux scraper une vraie source, dis-le-moi et je peux t'aider à écrire le script BeautifulSoup.

2. **Enrichir les autres fichiers `data/*.txt` si tu as le temps**
   (sources officielles OMS, Ministère de la Santé du Burkina Faso, etc.) pour rendre la base plus complète — sans copier de longs textes protégés, reformule les informations dans tes propres mots.

3. **Déployer l'application**
   Options gratuites : Render.com (recommandé pour FastAPI), Railway, ou Hugging Face Spaces (Docker). Dis-moi si tu veux le guide de déploiement.

4. **Rédiger le rapport technique** en réutilisant :
   - Le schéma d'architecture ci-dessus.
   - Les choix de méthodologie (chunking à 800 caractères avec 150 de chevauchement, embeddings `all-MiniLM-L6-v2`, LLM Llama 3.3 70B via Groq).
   - Les scores obtenus avec `evaluate.py`.
   - Une section "Limites" : base de connaissances encore limitée, pas de mise à jour temps réel des pharmacies de garde, pas de vérification médicale professionnelle des contenus.

5. **Pousser le code sur GitHub** (dépôt obligatoire) avec ce `README.md`.
