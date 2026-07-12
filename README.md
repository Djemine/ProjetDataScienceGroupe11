# Assistant d'Orientation Médicale & Prévention

Projet réalisé dans le cadre du Master 1 IFOAD — Data Science 2026 (Option 3 : Agent d'Orientation Médicale & Prévention).

L'idée de départ : au Burkina Faso, trouver une pharmacie de garde ou un centre de santé, ou simplement avoir une info fiable sur le paludisme, la dengue ou la nutrition, ce n'est pas toujours simple. Ce projet propose un assistant conversationnel qui répond à ces questions à partir d'une base documentaire vérifiée, complétée par une recherche en temps réel pour les pharmacies de garde (dont la liste change chaque jour).

## Comment ça marche

L'agent ne se contente pas d'interroger un modèle de langage à l'aveugle. À chaque question, il détermine d'abord de quoi on parle réellement — un centre de santé, une pharmacie de garde, ou un sujet de santé publique — en tenant compte du fil de la conversation (une question comme "et à Bobo ?" doit être comprise dans son contexte). Selon le cas, il va chercher l'information soit dans une base vectorielle constituée à partir de documents officiels, soit en scrapant en direct un site listant les pharmacies de garde du jour.

```
Question utilisateur
      │
      ▼
Résolution du sujet (pharmacie / centre de santé / thème santé)
      │
      ├── pharmacie de garde ──► scraping en direct (infossante.net)
      │
      └── sinon ──► recherche vectorielle (ChromaDB + sentence-transformers)
      │
      ▼
Contexte assemblé + historique de conversation
      │
      ▼
Génération de la réponse (Groq, Llama 3.3 70B) avec un prompt système
qui interdit les diagnostics, les dosages, et les réponses hors-base
      │
      ▼
Réponse + sources affichées
```

## Sources documentaires

- **Liste du Réseau Médical ASK Gras Savoye Burkina** (annuaire de cliniques, hôpitaux et pharmacies pour Ouagadougou, Bobo-Dioulasso et les provinces)
- **Guide des Diagnostics et Traitements** (GDT, mars 2024)
- **Guide SIMR** (Système Intégré de Surveillance et de Riposte, 2012) pour la définition de cas de la dengue
- Aide-mémoires de l'OMS (Bureau régional de la Méditerranée orientale) pour le paludisme et la dengue
- Fiches de nutrition et de conseils généraux rédigées par l'équipe

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Récupère une clé API gratuite sur [console.groq.com](https://console.groq.com), puis :

```bash
cp .env.example .env
# éditer .env et coller la clé : GROQ_API_KEY=gsk_xxxxx
```

## Lancer le projet

**1. Construire la base vectorielle** (à refaire si les fichiers dans `data/` changent) :
```bash
python ingest.py
```

**2. Démarrer le serveur :**
```bash
uvicorn main:app --reload
```
L'interface est accessible sur http://127.0.0.1:8000

**3. Évaluer la robustesse du système :**
```bash
python evaluate.py
```
Ce script mesure la pertinence de la recherche vectorielle, la fiabilité du routage conversationnel, le taux de refus sur les questions hors-sujet, et écrit un résumé dans `evaluation_results.md`, réutilisable tel quel dans le rapport technique.

## Structure du projet

```
agentRAGsante/
├── data/                       # Corpus documentaire (voir "Sources" ci-dessus)
├── vectordb/                   # Base ChromaDB générée par ingest.py
├── static/
│   └── index.html              # Interface web
├── ingest.py                   # Découpage + vectorisation du corpus
├── rag_engine.py                # Logique de l'agent : routage, recherche, génération
├── live_scraper.py              # Scraping des pharmacies de garde en temps réel
├── main.py                     # API FastAPI
├── evaluate.py                  # Tests de robustesse du système
├── Dockerfile                  # Déploiement (Render / Hugging Face Spaces)
├── requirements.txt
└── .env.example
```

## Déploiement

L'application est déployée sur Render : [agent-rag-santegroupe11.onrender.com](https://agent-rag-santegroupe11.onrender.com)

## Limites connues

- La détection de ville dans les questions repose sur une correspondance exacte : une faute de frappe sur un nom de ville n'est pas reconnue.
- Le respect strict du contexte documentaire par le modèle de langage n'est jamais garanti à 100 %, malgré les règles imposées dans le prompt système.
- L'API Groq utilisée est soumise à des quotas (par minute et par jour) sur le palier gratuit, ce qui peut occasionnellement limiter la disponibilité du service.
- Le corpus documentaire, bien que basé sur des sources officielles pour l'essentiel, reste d'un volume limité — il gagnerait à être enrichi.

## Équipe

Groupe 11 — Master 1 IFOAD, Data Science 2026