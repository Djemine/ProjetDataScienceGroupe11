"""
rag_engine.py
Recherche vectorielle dans ChromaDB + appel au LLM (Groq) avec le contexte récupéré.
"""

import os
import re
import time
import unicodedata
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq

from live_scraper import get_pharmacies_de_garde, format_for_llm, VILLE_ALIASES

load_dotenv()


GREETING_KEYWORDS = [
    "salut", "bonjour", "bonsoir", "coucou", "hello",
    "comment tu vas", "comment allez vous",
    "ca va",
]


def normalize_text(text: str) -> str:
    """Retire les accents et met en minuscule, pour comparer 'santé'/'sante', 'hôpital'/'hopital', etc."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


# Toutes les listes ci-dessous n'ont besoin que d'une seule forme (sans accent) :
# normalize_text() est appliqué des deux côtés au moment de la comparaison.

PHARMACY_KEYWORDS = ["pharmacie", "garde", "pharmacies"]

FACILITY_KEYWORDS = [
    "clinique", "cliniques",
    "centre de sante", "centres de sante",
    "centre medical", "centres medicaux",
    "hopital", "hopitaux",
    "csps", "cma", "chu", "chr",
    "dispensaire", "dispensaires",
    "polyclinique", "polycliniques",
    "cabinet medical", "cabinet dentaire",
    "maternite", "maternites",
    "laboratoire", "laboratoires",
]

HEALTH_TOPIC_KEYWORDS = [
    "paludisme", "dengue", "nutrition", "grossesse", "grossesses",
    "diabete", "tension", "hypertension", "vaccin", "vaccins", "allaitement",
    "enfant", "enfants", "bebe", "bebes",
    "accouchement", "planning familial", "hygiene",
]

# Tout ce qui doit empêcher un routage vers le scraping pharmacie
NON_PHARMACY_KEYWORDS = FACILITY_KEYWORDS + HEALTH_TOPIC_KEYWORDS

# Sous-ensemble qui déclenche le filtrage par ville sur les fichiers de centres de santé
CENTRE_KEYWORDS = FACILITY_KEYWORDS

DB_DIR = "vectordb"
COLLECTION_NAME = "sante_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 8
CENTRE_MERGE_CAP = 12          # nb de chunks gardés une fois le filtrage par ville appliqué
CONTINUITY_WORD_LIMIT = 15     # au-delà, une question est traitée comme un nouveau sujet

CITY_SOURCE_FILES = {
    "ouagadougou": "centres_sante_ouagadougou.txt",
    "bobo-dioulasso": "centres_sante_bobo.txt",
}
PROVINCE_SOURCE_FILE = "centres_sante_province.txt"

# --- Garde-fous de volumétrie (budget TPM Groq) ---
MAX_HISTORY_MESSAGES = 6        # ~3 échanges user/assistant récents, suffisant pour la continuité
MAX_CONTEXT_CHARS = 4000        # cap du contexte RAG injecté, quel que soit le nombre de chunks
MAX_TOKENS_ANSWER = 1500        # équilibre troncature / budget TPM disponible

SYSTEM_PROMPT = """Tu es un assistant d'orientation et de prévention santé de premier niveau, destiné au grand public au Burkina Faso.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des informations fournies dans le "CONTEXTE" ci-dessous.
2. Si le contexte contient des informations pertinentes (même partielles) pour répondre à la question, utilise-les et réponds. Dis "Je n'ai pas cette information dans ma base de connaissances" UNIQUEMENT si le contexte ne contient RÉELLEMENT aucune information liée au sujet de la question — ne le dis jamais si le contexte contient des éléments pertinents, même si la liste n'est pas exhaustive.
3. Ne donne JAMAIS de diagnostic médical définitif. Si la question porte sur un symptôme, une maladie ou une orientation de soin, rappelle qu'il s'agit d'une orientation de premier niveau, pas d'une consultation médicale. N'ajoute PAS ce rappel pour une question purement factuelle (voir règle 6).
4. Le contexte peut contenir des extraits de protocoles cliniques officiels destinés aux agents de santé (définitions de cas, signes de gravité, posologies, noms de médicaments). Tu peux utiliser les définitions de cas et signes de gravité pour orienter l'utilisateur, mais NE CITE JAMAIS le nom d'un médicament, d'une molécule, ou d'une classe thérapeutique (ex: "quinine", "ACT", "paracétamol"), et NE DONNE JAMAIS de dosage ou de protocole de traitement détaillé : cela doit être décidé par un professionnel de santé après examen. Dis plutôt : "le choix du traitement doit être déterminé par un professionnel de santé au centre de santé, qui pourra vous prescrire ce qui est adapté à votre cas."
5. Pour les pharmacies de garde : ne mentionne JAMAIS de distance en kilomètres (ex: "à 1,14 km"), car cette donnée n'est pas fiable pour la position réelle de l'utilisateur. En revanche, si un lien "Itinéraire" (URL Google Maps) est présent dans le contexte, tu PEUX et DOIS le partager tel quel avec l'utilisateur : ce lien ne contient que la destination, Google Maps calculera automatiquement le trajet depuis la position réelle de la personne quand elle l'ouvre sur son propre appareil. Si le contexte liste PLUSIEURS pharmacies et que l'utilisateur demande une autre option (par exemple parce que la précédente lui semble trop loin, ou qu'il en veut une différente), propose-lui une autre pharmacie de la liste (nom, contact, lien Itinéraire) SANS te prononcer sur laquelle est la plus proche : tu n'as pas cette information, laisse l'utilisateur comparer lui-même via les liens Itinéraire. Ne dis JAMAIS qu'il n'y a pas d'autre option si le contexte contient plusieurs pharmacies.
6. Le disclaimer d'urgence ("rendez-vous au centre de santé le plus proche ou appelez les urgences") ne doit apparaître QUE si l'utilisateur décrit lui-même un symptôme ou une urgence dans sa question actuelle (ex: convulsions, difficultés respiratoires, saignements, perte de connaissance, forte fièvre). Pour toute question purement factuelle sans mention de symptôme (adresse, numéro de téléphone, pharmacie de garde, horaires, itinéraire), ne mentionne AUCUN disclaimer d'urgence ni rappel de type "orientation de premier niveau" : réponds simplement à la question posée.
7. Sois clair, empathique, et utilise un langage simple et accessible.
8. Réponds en français, sauf si l'utilisateur écrit dans une autre langue.
9. Si l'utilisateur redemande plusieurs fois "une autre" pharmacie, regarde l'historique de la conversation pour voir lesquelles tu as déjà proposées, et propose-en systématiquement une NOUVELLE de la liste, différente des précédentes. Si tu as déjà proposé toutes les pharmacies disponibles dans le contexte, dis-le honnêtement ("J'ai fait le tour des pharmacies de garde que j'ai trouvées pour aujourd'hui à [ville]") plutôt que de répéter une pharmacie déjà donnée.
10. Ne mentionne JAMAIS les noms techniques de fichiers, d'annuaires ou de la base de connaissances (ex: "centres_sante_bobo.txt", "le fichier", "l'annuaire nommé...") dans ta réponse. Présente l'information directement, comme si tu la connaissais, sans faire référence à sa source technique interne — les sources sont déjà affichées séparément à l'utilisateur par l'interface.
"""


def _all_city_terms():
    terms = []
    for ville, aliases in VILLE_ALIASES.items():
        terms.extend([ville, ville.replace("-", " ")] + aliases)
    return terms


# --- Détection par mots entiers (évite les faux positifs de sous-chaîne, ex: "garde" ⊂ "regardez") ---

def _compile_keyword_pattern(keywords):
    """Compile une regex à limites de mots pour un ensemble de mots-clés (accents déjà retirés
    en amont par normalize_text). Trie par longueur décroissante pour que les expressions
    composées ('centre de sante') soient testées avant leurs sous-mots."""
    escaped = sorted((re.escape(kw) for kw in keywords), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b")


PHARMACY_PATTERN = _compile_keyword_pattern(PHARMACY_KEYWORDS)
CENTRE_PATTERN = _compile_keyword_pattern(CENTRE_KEYWORDS)          # = FACILITY_KEYWORDS
NON_PHARMACY_PATTERN = _compile_keyword_pattern(NON_PHARMACY_KEYWORDS)
HEALTH_TOPIC_PATTERN = _compile_keyword_pattern(HEALTH_TOPIC_KEYWORDS)
PHARMACY_WORD_PATTERN = _compile_keyword_pattern(["pharmacie", "pharmacies"])
GARDE_PATTERN = _compile_keyword_pattern(["garde"])

TOPIC_PHARMACY = "pharmacy"
TOPIC_CENTRE = "centre"

GREETING_PATTERN = _compile_keyword_pattern(GREETING_KEYWORDS)


def _strip_greetings(query_norm: str) -> str:
    """Attend une chaîne déjà normalisée (normalize_text). Retire les salutations avant la
    résolution de sujet et la recherche vectorielle : sans ça, une question du type 'Salut,
    comment tu vas ?? Quels sont les symptômes du palu ?' se découpe (via le split sur '?')
    en une sous-question parasite ('comment tu vas') qui pollue le contexte récupéré avec des
    résultats hors sujet. Le LLM reçoit toujours `query` en clair dans le prompt final, pour
    répondre avec un ton naturel à la salutation."""
    cleaned = GREETING_PATTERN.sub("", query_norm)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ?!.,")
    return cleaned if cleaned else query_norm


def _last_explicit_topic(history):
    """Remonte l'historique EN NE CONSIDÉRANT QUE LES MESSAGES UTILISATEUR, du plus récent
    au plus ancien, et renvoie le premier sujet explicite rencontré. On ignore volontairement
    les messages de l'assistant : ses réponses peuvent contenir des mots comme "pharmacie" en
    tant que simple repère géographique (ex: "près de la pharmacie Sotisse"), ce qui ne reflète
    en rien l'intention de l'utilisateur et provoquerait un changement de sujet non désiré.
    La remontée n'est pas limitée à une fenêtre fixe : elle gère donc naturellement les chaînes
    de questions elliptiques ('Et à Bobo ?', 'Et à Banfora ?') quelle que soit leur longueur."""
    if not history:
        return None
    for m in reversed(history):
        if m.get("role") != "user":
            continue
        content_norm = normalize_text(m.get("content", ""))
        if PHARMACY_PATTERN.search(content_norm):
            return TOPIC_PHARMACY
        if CENTRE_PATTERN.search(content_norm):
            return TOPIC_CENTRE
    return None


def _trim_history(history):
    """Ne garde que les derniers échanges pour l'appel Groq (résolution de sujet, filtrage
    ville, etc. reçoivent toujours le history complet séparément). Évite de faire grossir
    indéfiniment chaque requête avec toute la session, ce qui dépasse vite le budget TPM."""
    if not history:
        return history
    return history[-MAX_HISTORY_MESSAGES:]


def _truncate_context(context_text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Plafonne la taille du contexte RAG injecté dans le prompt. Coupe proprement sur un
    séparateur de chunk plutôt qu'en plein milieu d'un bloc [Source: ...]."""
    if len(context_text) <= max_chars:
        return context_text
    truncated = context_text[:max_chars]
    last_sep = truncated.rfind("\n\n---\n\n")
    if last_sep > 0:
        truncated = truncated[:last_sep]
    return truncated + "\n\n[...contexte tronqué pour respecter la limite de taille...]"


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY manquant. Crée un fichier .env avec GROQ_API_KEY=ta_cle "
                "(clé gratuite sur https://console.groq.com)"
            )
        self.groq_client = Groq(api_key=api_key)

        chroma_client = chromadb.PersistentClient(path=DB_DIR)
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        try:
            self.collection = chroma_client.get_collection(
                name=COLLECTION_NAME, embedding_function=embedding_fn
            )
        except Exception as e:
            raise RuntimeError(
                "Collection ChromaDB introuvable. As-tu lancé 'python ingest.py' d'abord ?"
            ) from e

    def retrieve(self, query: str, top_k: int = TOP_K, where=None):
        results = self.collection.query(query_texts=[query], n_results=top_k, where=where)

        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {"text": doc, "source": meta.get("source", "inconnu"), "score": 1 - dist}
            for doc, meta, dist in zip(docs, metadatas, distances)
        ]

    def _build_retrieval_query(self, query: str, history=None) -> str:
        """Enrichit une question courte et ambiguë avec la question précédente, pour que la
        recherche vectorielle ne perde pas le fil du sujet (ex: 'quelles mesures alors ?').
        Ne s'applique JAMAIS si la question porte déjà, à elle seule, un sujet reconnaissable
        (ex: 'Nutrition pendant la grossesse') : dans ce cas, l'enrichir avec le tour précédent
        risquerait au contraire de polluer la recherche avec un sujet sans rapport."""
        if not history or len(query.split()) > CONTINUITY_WORD_LIMIT:
            return query

        query_norm = normalize_text(query)
        if HEALTH_TOPIC_PATTERN.search(query_norm) or CENTRE_PATTERN.search(query_norm):
            return query

        last_user_message = ""
        for m in reversed(history):
            if m.get("role") == "user":
                last_user_message = m.get("content", "")
                break

        if last_user_message:
            return f"{last_user_message} {query}"
        return query
    
    def _detect_city(self, query: str, history=None):
        """Cherche une ville dans la question, puis dans l'historique récent si absente."""
        query_norm = normalize_text(query)
        for ville, aliases in VILLE_ALIASES.items():
            terms = [ville, ville.replace("-", " ")] + aliases
            if any(normalize_text(t) in query_norm for t in terms):
                return ville

        if history:
            for m in reversed(history):
                content_norm = normalize_text(m.get("content", ""))
                for ville, aliases in VILLE_ALIASES.items():
                    terms = [ville, ville.replace("-", " ")] + aliases
                    if any(normalize_text(t) in content_norm for t in terms):
                        return ville

        return None

    def _build_pharmacy_query(self, query: str, history=None) -> str:
        """Si la question ne mentionne aucune ville (ex: 'une autre ?'), récupère la dernière
        ville évoquée dans l'historique plutôt que de retomber sur Ouagadougou par défaut."""
        city_terms = _all_city_terms()
        query_norm = normalize_text(query)

        if any(normalize_text(t) in query_norm for t in city_terms):
            return query

        if history:
            for m in reversed(history):
                content_norm = normalize_text(m.get("content", ""))
                if any(normalize_text(t) in content_norm for t in city_terms):
                    return m.get("content", "") + " " + query

        return query

    def _resolve_topic(self, query: str, history=None):
        """Point d'entrée UNIQUE pour déterminer le sujet de la question courante
        (pharmacie / centre de santé / aucun). is_pharmacy_query() et _is_centre_query()
        s'appuient tous les deux sur cette même résolution : il n'existe donc plus d'ordre
        d'appel implicite qui puisse faire gagner un sujet sur l'autre par accident.
        """
        query_norm = normalize_text(query)

        # Thème de santé pur (paludisme, nutrition...) : jamais lié à une pharmacie,
        # et ne déclenche pas non plus le filtrage ville des centres.
        if HEALTH_TOPIC_PATTERN.search(query_norm):
            return None

        has_facility = CENTRE_PATTERN.search(query_norm)
        has_pharmacy_word = PHARMACY_WORD_PATTERN.search(query_norm)
        has_garde = GARDE_PATTERN.search(query_norm)

        # Signal fort et sans ambiguïté : "pharmacie" + "garde" ensemble = intention de
        # disponibilité temps réel, même si un lieu (hôpital, clinique...) est mentionné
        # comme simple complément ("la pharmacie DE L'HÔPITAL est-elle de garde ?").
        strong_pharmacy_intent = bool(has_pharmacy_word and has_garde)

        if has_facility and not strong_pharmacy_intent:
            return TOPIC_CENTRE

        if has_pharmacy_word or has_garde:
            return TOPIC_PHARMACY

        # Question courte sans mot-clé explicite : probablement elliptique -> héritage du
        # dernier sujet explicite exprimé par l'utilisateur (jamais déclenché par une simple
        # ville, et jamais par un mot apparu incidemment dans une réponse de l'assistant).
        if len(query_norm.split()) <= CONTINUITY_WORD_LIMIT:
            return _last_explicit_topic(history)

        return None

    def is_pharmacy_query(self, query: str, history=None) -> bool:
        return self._resolve_topic(query, history) == TOPIC_PHARMACY

    def _is_centre_query(self, query: str, history=None, topic=None) -> bool:
        if topic is None:
            topic = self._resolve_topic(query, history)
        return topic == TOPIC_CENTRE

    def _retrieve_for_centre_query(self, query: str, history, retrieved_chunks, topic=None):
        """Filtre/complète la recherche vectorielle par métadonnées quand une ville est connue,
        car la seule similarité sémantique ne garantit pas de rester sur le bon fichier ville.
        Exclut explicitement les fichiers des AUTRES villes du remplissage complémentaire,
        pour éviter qu'une réponse sur Ouagadougou n'inclue des centres de Bobo (ou inversement)
        simplement parce que le filtrage ciblé n'a pas suffi à remplir CENTRE_MERGE_CAP."""
        if not self._is_centre_query(query, history, topic=topic):
            return retrieved_chunks

        city = self._detect_city(query, history)
        if city is None:
            return retrieved_chunks

        target_file = CITY_SOURCE_FILES.get(city, PROVINCE_SOURCE_FILE)
        other_city_files = set(CITY_SOURCE_FILES.values()) - {target_file}

        # top_k large : les fichiers par ville sont courts (quelques dizaines de chunks max),
        # on peut se permettre de les couvrir presque intégralement plutôt que de risquer
        # de rater une section (ex: dentaire, cardiologie) faute de proximité sémantique.
        targeted = self.retrieve(query, top_k=20, where={"source": target_file})

        seen_texts = {c["text"] for c in targeted}
        filler = [
            c for c in retrieved_chunks
            if c["text"] not in seen_texts and c["source"] not in other_city_files
        ]
        merged = targeted + filler
        return merged[:CENTRE_MERGE_CAP]

    def generate_answer(self, query: str, history=None):
        retrieval_start = time.time()

        # search_query : version normalisée ET débarrassée des salutations, utilisée pour
        # TOUTE la résolution de sujet / recherche. `query` (original, avec la salutation)
        # n'est réinjecté qu'au moment de construire le prompt final envoyé au LLM.
        search_query = _strip_greetings(normalize_text(query))
        topic = self._resolve_topic(search_query, history)

        if topic == TOPIC_PHARMACY:
            scraping_query = self._build_pharmacy_query(search_query, history=history)
            live_result = get_pharmacies_de_garde(scraping_query)
            context_text = format_for_llm(live_result)
            sources_meta = [{"source": "infossante.net (temps réel)", "score": 1.0}]
            retrieval_mode = "scraping_live"
        else:
            sub_questions = [q.strip() for q in search_query.split("?") if q.strip()]

            if len(sub_questions) > 1:
                seen_texts = set()
                retrieved_chunks = []
                for sq in sub_questions:
                    for c in self.retrieve(sq):
                        if c["text"] not in seen_texts:
                            seen_texts.add(c["text"])
                            retrieved_chunks.append(c)
            else:
                retrieval_query = self._build_retrieval_query(search_query, history)
                retrieved_chunks = self.retrieve(retrieval_query)

            retrieved_chunks = self._retrieve_for_centre_query(search_query, history, retrieved_chunks, topic=topic)
            context_text = "\n\n---\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in retrieved_chunks
            )

            best_score_by_source = {}
            for c in retrieved_chunks:
                src = c["source"]
                if src not in best_score_by_source or c["score"] > best_score_by_source[src]:
                    best_score_by_source[src] = c["score"]

            sources_meta = [
                {"source": src, "score": round(score, 3)}
                for src, score in sorted(best_score_by_source.items(), key=lambda x: -x[1])
            ]
            retrieval_mode = "recherche_vectorielle"

        retrieval_ms = round((time.time() - retrieval_start) * 1000)

        # --- Construction du prompt final, avec plafonnement de la taille envoyée à Groq ---
        context_text = _truncate_context(context_text)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        trimmed_history = _trim_history(history)
        if trimmed_history:
            messages.extend(trimmed_history)

        user_message = f"""CONTEXTE (extrait de la base de connaissances) :
{context_text}

QUESTION DE L'UTILISATEUR :
{query}"""

        messages.append({"role": "user", "content": user_message})

        generation_start = time.time()
        completion = self.groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_TOKENS_ANSWER,
        )
        generation_ms = round((time.time() - generation_start) * 1000)

        answer = completion.choices[0].message.content
        was_truncated = completion.choices[0].finish_reason == "length"
        if was_truncated:
            answer += "\n\n*(Réponse tronquée par manque de place — demande-moi de continuer si besoin.)*"

        return {
            "answer": answer,
            "sources": sources_meta,
            "analysis": {
                "retrieval_mode": retrieval_mode,
                "chunks_retrieved": len(sources_meta),
                "top_k": TOP_K,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "model": GROQ_MODEL,
                "truncated": was_truncated,
            },
        }