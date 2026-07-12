# Résultats d'évaluation — 11/07/2026 22:34

Ce fichier est généré automatiquement par `evaluate.py`. Les scores ci-dessous sont directement réutilisables dans la section « Expérimentation et évaluation » du rapport technique.

## 1. Pertinence de la recherche vectorielle
- Précision top-1 : **71%** (5/7)
- MRR (top-3) : **0.786**

## 2. Robustesse du routage conversationnel
- Score global : **100%** (16/16 assertions)

Détail par scénario :
- ✅ Chaîne d'ellipses sur les centres de santé (villes successives)
- ✅ Bascule volontaire centre -> pharmacie puis retour ville
- ✅ Question indépendante sur un thème de santé (ne doit hériter d'aucun sujet)
- ✅ Ambiguïté lexicale pharmacie / établissement
- ✅ Mention d'un établissement sans signal de garde (reste centre)
- ✅ Ville seule en première question (ne doit jamais déclencher la pharmacie par défaut)
- ✅ Faux positif lexical 'garde' dans une réponse assistant (ne doit pas changer le sujet)

## 3. Robustesse de la détection de ville (informatif)
- Taux de détection avec fautes de frappe incluses : **75%** (3/4)
- Limite connue : correspondance exacte, non tolérante aux fautes de frappe (cf. section 10.1 du rapport).

## 4. Taux de refus correct (anti-hallucination)
- Score : **100%** (6/6)

## 5. Routage bout en bout (generate_answer)
- Score : **100%** (2/2)

### Temps de réponse mesurés
| Question | Mode | Recherche (ms) | Génération (ms) |
|---|---|---|---|
| Quels sont les symptômes du paludisme chez l'enfan... | recherche_vectorielle | 11 | 15998 |
| Quelle pharmacie est de garde à Ouagadougou ce soi... | scraping_live | 1033 | 17388 |

---
*Rapport généré automatiquement — à joindre en annexe du rapport technique.*