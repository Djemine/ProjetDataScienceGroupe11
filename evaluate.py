"""
evaluate.py
Étape 4 du projet : évaluation de la robustesse du système RAG.

Ce script mesure quatre dimensions distinctes :

1. Pertinence de la recherche vectorielle (précision top-1 et MRR sur top-3)
2. Robustesse du routage conversationnel (pharmacie / centre de santé / aucun sujet),
   y compris sur des cas de continuité, d'ambiguïté lexicale et de fautes de frappe
3. Taux de refus correct sur des questions hors-sujet (anti-hallucination)
4. Test de bout en bout (generate_answer) sur des cas représentatifs, pour vérifier
   que le mode de récupération effectivement utilisé correspond à l'attendu

Usage : python evaluate.py

À la fin de l'exécution, un fichier evaluation_results.md est généré à la racine
du projet : il contient un résumé chiffré directement réutilisable dans la section
"Expérimentation et évaluation" du rapport technique.
"""

import time
from datetime import datetime

from rag_engine import RAGEngine, TOPIC_PHARMACY, TOPIC_CENTRE


# =====================================================================
# 1. JEU DE TEST : PERTINENCE DE LA RECHERCHE VECTORIELLE
# =====================================================================
# (question, source attendue) — la source attendue doit correspondre exactement
# au nom de fichier tel qu'il apparaît dans data/ (voir metadata "source" de ChromaDB)
RELEVANCE_TEST_CASES = [
    ("Quels sont les symptômes du paludisme ?", "paludisme.txt"),
    ("Comment prévenir la dengue ?", "dengue.txt"),
    ("Quels aliments donner à un enfant de moins de 5 ans ?", "nutrition.txt"),
    ("Quelle alimentation recommandée pendant la grossesse ?", "sante_mere_enfant.txt"),
    ("Quelle est la définition de cas de la dengue en surveillance épidémiologique ?", "simr_dengue_definition_cas.txt"),
    ("Quels sont les conseils d'hygiène de base à respecter au quotidien ?", "conseils_generaux_sante.txt"),
    ("Quel est le protocole prioritaire de prise en charge du paludisme grave ?", "gdt_protocoles_prioritaires.txt"),
]


# =====================================================================
# 2. JEU DE TEST : ROUTAGE CONVERSATIONNEL
# =====================================================================
# Chaque scénario est une liste de tours : (question, sujet_attendu)
# sujet_attendu ∈ {TOPIC_PHARMACY, TOPIC_CENTRE, None}
# L'historique est reconstruit tour après tour, comme en conditions réelles.

ROUTING_SCENARIOS = {
    "Chaîne d'ellipses sur les centres de santé (villes successives)": [
        ("Connais-tu des centres de santé à Ouagadougou ?", TOPIC_CENTRE),
        ("Et à Bobo ?", TOPIC_CENTRE),
        ("Et à Banfora ?", TOPIC_CENTRE),
        ("Et là-bas, il y a un CSPS ?", TOPIC_CENTRE),
    ],
    "Bascule volontaire centre -> pharmacie puis retour ville": [
        ("Centres de santé à Ouagadougou ?", TOPIC_CENTRE),
        ("Et les pharmacies de garde ?", TOPIC_PHARMACY),
        ("Une autre ?", TOPIC_PHARMACY),
        ("Et à Bobo ?", TOPIC_PHARMACY),
    ],
    "Question indépendante sur un thème de santé (ne doit hériter d'aucun sujet)": [
        ("Centres de santé à Ouagadougou ?", TOPIC_CENTRE),
        ("Quels sont les signes du paludisme ?", None),
        ("Et la dengue ?", None),
    ],
    "Ambiguïté lexicale pharmacie / établissement": [
        ("Est-ce que la pharmacie de l'hôpital est de garde ce soir ?", TOPIC_PHARMACY),
    ],
    "Mention d'un établissement sans signal de garde (reste centre)": [
        ("Où se trouve la pharmacie de l'hôpital Yalgado ?", TOPIC_CENTRE),
    ],
    "Ville seule en première question (ne doit jamais déclencher la pharmacie par défaut)": [
        ("Ouagadougou", None),
    ],
    "Faux positif lexical 'garde' dans une réponse assistant (ne doit pas changer le sujet)": [
        ("Centres de santé à Ouagadougou ?", TOPIC_CENTRE),
        ("__ASSISTANT__:Voici les centres, gardez ce numéro et regardez ce lien.", None),
        ("Et à Bobo ?", TOPIC_CENTRE),
    ],
}


# =====================================================================
# 3. JEU DE TEST : ROBUSTESSE AUX FAUTES DE FRAPPE SUR LES VILLES
# =====================================================================
CITY_TYPO_TEST_CASES = [
    ("Koudougou", "koudougou"),
    ("Kougougou", "koudougou"),   
    ("Bobo", "bobo-dioulasso"),
    ("ouaga", "ouagadougou"),
]


# =====================================================================
# 4. JEU DE TEST : ANTI-HALLUCINATION (questions hors du domaine du corpus)
# =====================================================================
HALLUCINATION_TEST_CASES = [
    "Quel est le prix du Doliprane en pharmacie cette semaine ?",
    "Peux-tu me donner la recette du tô traditionnel étape par étape ?",
    "Quel est le taux de change actuel du FCFA en euros ?",
    "Qui a gagné le dernier match de la CAN ?",
    "Quelle est la météo à Ouagadougou demain ?",
    "Peux-tu m'aider à réserver un billet d'avion pour Paris ?",
]

REFUSAL_KEYWORDS = [
    "je n'ai pas cette information",
    "ne dispose pas",
    "n'est pas dans ma base",
    "consulter un professionnel",
    "je ne sais pas",
    "aucune information",
    "impossible de récupérer",
]


# =====================================================================
# 5. JEU DE TEST : BOUT EN BOUT (generate_answer complet)
# =====================================================================
END_TO_END_TEST_CASES = [
    ("Quels sont les symptômes du paludisme chez l'enfant ?", "recherche_vectorielle"),
    ("Quelle pharmacie est de garde à Ouagadougou ce soir ?", "scraping_live"),
]


# =====================================================================
# Fonctions de test
# =====================================================================

def test_relevance(engine: RAGEngine, top_k: int = 3):
    print("\n" + "=" * 70)
    print("TEST 1 — PERTINENCE DE LA RECHERCHE VECTORIELLE")
    print("=" * 70)

    top1_correct = 0
    reciprocal_ranks = []
    details = []

    for question, expected_source in RELEVANCE_TEST_CASES:
        retrieved = engine.retrieve(question, top_k=top_k)
        sources = [c["source"] for c in retrieved]

        top1_ok = bool(sources) and sources[0] == expected_source
        top1_correct += top1_ok

        if expected_source in sources:
            rank = sources.index(expected_source) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            rank = None
            reciprocal_ranks.append(0.0)

        status = "OK" if top1_ok else ("PARTIEL" if rank else "ECHEC")
        top_score = retrieved[0]["score"] if retrieved else 0.0
        print(f"[{status}] {question}")
        print(f"        Attendu : {expected_source} | Top-1 obtenu : {sources[0] if sources else 'aucun'} "
              f"(score={top_score:.3f}) | Rang de la source attendue : {rank or '>' + str(top_k)}")

        details.append({
            "question": question, "expected": expected_source,
            "top1": sources[0] if sources else None, "rank": rank, "status": status,
        })

    n = len(RELEVANCE_TEST_CASES)
    top1_score = top1_correct / n * 100
    mrr = sum(reciprocal_ranks) / n

    print(f"\nPrécision top-1 : {top1_correct}/{n} ({top1_score:.0f}%)")
    print(f"MRR (top-{top_k})  : {mrr:.3f}")
    return {"top1_score": top1_score, "mrr": mrr, "n": n, "details": details}


def test_routing(engine: RAGEngine):
    print("\n" + "=" * 70)
    print("TEST 2 — ROBUSTESSE DU ROUTAGE CONVERSATIONNEL")
    print("=" * 70)

    total, correct = 0, 0
    scenario_results = []

    for scenario_name, turns in ROUTING_SCENARIOS.items():
        print(f"\n--- Scénario : {scenario_name} ---")
        history = []
        scenario_ok = True

        for question, expected_topic in turns:
            if question.startswith("__ASSISTANT__:"):
                history.append({"role": "assistant", "content": question[len("__ASSISTANT__:"):]})
                continue

            topic = engine._resolve_topic(question, history=history)
            ok = topic == expected_topic
            total += 1
            correct += ok
            scenario_ok = scenario_ok and ok

            status = "OK" if ok else "ECHEC"
            print(f"  [{status}] \"{question}\" -> obtenu={topic!r} attendu={expected_topic!r}")

            history.append({"role": "user", "content": question})
            # réponse assistant simulée neutre, pour faire progresser l'historique
            # comme en conditions réelles sans introduire de mot-clé parasite
            history.append({"role": "assistant", "content": "Voici les informations correspondantes."})

        scenario_results.append({"name": scenario_name, "ok": scenario_ok})

    score = correct / total * 100 if total else 0
    print(f"\nScore global de routage : {correct}/{total} ({score:.0f}%)")
    return {"score": score, "correct": correct, "total": total, "scenarios": scenario_results}


def test_city_typo_robustness(engine: RAGEngine):
    print("\n" + "=" * 70)
    print("TEST 3 — ROBUSTESSE DE LA DÉTECTION DE VILLE (informatif)")
    print("=" * 70)
    print("Ce test documente une limite connue (section 10.1 du rapport) : la détection")
    print("de ville repose sur une correspondance exacte, sans tolérance aux fautes de")
    print("frappe. Il n'est pas comptabilisé dans le score global de robustesse.\n")

    correct = 0
    for input_text, expected_city in CITY_TYPO_TEST_CASES:
        detected = engine._detect_city(input_text)
        ok = detected == expected_city
        correct += ok
        status = "OK" if ok else "NON DETECTE (limite connue)"
        print(f"  [{status}] \"{input_text}\" -> détecté={detected!r} attendu={expected_city!r}")

    score = correct / len(CITY_TYPO_TEST_CASES) * 100
    print(f"\nTaux de détection (avec fautes de frappe incluses) : {correct}/{len(CITY_TYPO_TEST_CASES)} ({score:.0f}%)")
    return {"score": score, "n": len(CITY_TYPO_TEST_CASES)}


def test_hallucination(engine: RAGEngine):
    print("\n" + "=" * 70)
    print("TEST 4 — TAUX D'HALLUCINATION (questions hors-sujet)")
    print("=" * 70)

    correct_refusals = 0
    details = []

    for question in HALLUCINATION_TEST_CASES:
        result = engine.generate_answer(question)
        answer_lower = result["answer"].lower()
        refused = any(kw in answer_lower for kw in REFUSAL_KEYWORDS)
        correct_refusals += refused

        status = "OK (a refusé)" if refused else "ECHEC (a peut-être halluciné)"
        print(f"[{status}] {question}")
        print(f"        Réponse : {result['answer'][:180]}...\n")

        details.append({"question": question, "refused": refused, "answer_excerpt": result["answer"][:180]})

    n = len(HALLUCINATION_TEST_CASES)
    score = correct_refusals / n * 100
    print(f"Taux de refus correct : {correct_refusals}/{n} ({score:.0f}%)")
    return {"score": score, "n": n, "details": details}


def test_end_to_end(engine: RAGEngine):
    print("\n" + "=" * 70)
    print("TEST 5 — BOUT EN BOUT (generate_answer)")
    print("=" * 70)

    correct = 0
    details = []

    for question, expected_mode in END_TO_END_TEST_CASES:
        start = time.time()
        result = engine.generate_answer(question)
        elapsed_ms = round((time.time() - start) * 1000)

        actual_mode = result["analysis"]["retrieval_mode"]
        ok = actual_mode == expected_mode
        correct += ok

        status = "OK" if ok else "ECHEC"
        print(f"[{status}] {question}")
        print(f"        Mode attendu : {expected_mode} | Mode obtenu : {actual_mode}")
        print(f"        Chunks/sources : {result['analysis']['chunks_retrieved']} | "
              f"Temps recherche : {result['analysis']['retrieval_ms']} ms | "
              f"Temps génération : {result['analysis']['generation_ms']} ms | "
              f"Temps total mesuré : {elapsed_ms} ms | "
              f"Tronqué : {result['analysis'].get('truncated', 'N/A')}\n")

        details.append({
            "question": question, "expected_mode": expected_mode, "actual_mode": actual_mode,
            "retrieval_ms": result["analysis"]["retrieval_ms"],
            "generation_ms": result["analysis"]["generation_ms"],
        })

    n = len(END_TO_END_TEST_CASES)
    score = correct / n * 100
    print(f"Score de routage bout en bout : {correct}/{n} ({score:.0f}%)")
    return {"score": score, "n": n, "details": details}


# =====================================================================
# Rapport final
# =====================================================================

def write_report(results: dict, filepath: str = "evaluation_results.md"):
    now = datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = [
        f"# Résultats d'évaluation — {now}",
        "",
        "Ce fichier est généré automatiquement par `evaluate.py`. "
        "Les scores ci-dessous sont directement réutilisables dans la section "
        "« Expérimentation et évaluation » du rapport technique.",
        "",
        "## 1. Pertinence de la recherche vectorielle",
        f"- Précision top-1 : **{results['relevance']['top1_score']:.0f}%** "
        f"({results['relevance']['top1_score']/100*results['relevance']['n']:.0f}/{results['relevance']['n']})",
        f"- MRR (top-3) : **{results['relevance']['mrr']:.3f}**",
        "",
        "## 2. Robustesse du routage conversationnel",
        f"- Score global : **{results['routing']['score']:.0f}%** "
        f"({results['routing']['correct']}/{results['routing']['total']} assertions)",
        "",
        "Détail par scénario :",
    ]
    for s in results["routing"]["scenarios"]:
        mark = "✅" if s["ok"] else "❌"
        lines.append(f"- {mark} {s['name']}")

    lines += [
        "",
        "## 3. Robustesse de la détection de ville (informatif)",
        f"- Taux de détection avec fautes de frappe incluses : **{results['city']['score']:.0f}%** "
        f"({results['city']['score']/100*results['city']['n']:.0f}/{results['city']['n']})",
        "- Limite connue : correspondance exacte, non tolérante aux fautes de frappe "
        "(cf. section 10.1 du rapport).",
        "",
        "## 4. Taux de refus correct (anti-hallucination)",
        f"- Score : **{results['hallucination']['score']:.0f}%** "
        f"({results['hallucination']['score']/100*results['hallucination']['n']:.0f}/{results['hallucination']['n']})",
        "",
        "## 5. Routage bout en bout (generate_answer)",
        f"- Score : **{results['end_to_end']['score']:.0f}%** "
        f"({results['end_to_end']['score']/100*results['end_to_end']['n']:.0f}/{results['end_to_end']['n']})",
        "",
        "### Temps de réponse mesurés",
        "| Question | Mode | Recherche (ms) | Génération (ms) |",
        "|---|---|---|---|",
    ]
    for d in results["end_to_end"]["details"]:
        lines.append(f"| {d['question'][:50]}... | {d['actual_mode']} | {d['retrieval_ms']} | {d['generation_ms']} |")

    lines += [
        "",
        "---",
        "*Rapport généré automatiquement — à joindre en annexe du rapport technique.*",
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nRapport détaillé écrit dans : {filepath}")


if __name__ == "__main__":
    print("Initialisation du moteur RAG...")
    engine = RAGEngine()

    results = {}
    results["relevance"] = test_relevance(engine)
    results["routing"] = test_routing(engine)
    results["city"] = test_city_typo_robustness(engine)
    results["hallucination"] = test_hallucination(engine)
    results["end_to_end"] = test_end_to_end(engine)

    print("\n" + "=" * 70)
    print("RÉSUMÉ POUR LE RAPPORT TECHNIQUE")
    print("=" * 70)
    print(f"Pertinence recherche vectorielle (top-1)      : {results['relevance']['top1_score']:.0f}%")
    print(f"Pertinence recherche vectorielle (MRR top-3)  : {results['relevance']['mrr']:.3f}")
    print(f"Robustesse du routage conversationnel         : {results['routing']['score']:.0f}%")
    print(f"Détection de ville (avec fautes de frappe)    : {results['city']['score']:.0f}% (informatif)")
    print(f"Taux de refus correct (anti-hallucination)    : {results['hallucination']['score']:.0f}%")
    print(f"Routage bout en bout (generate_answer)        : {results['end_to_end']['score']:.0f}%")

    write_report(results)