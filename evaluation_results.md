# Resultats d'evaluation -- 12/07/2026 08:56

Ce fichier est genere automatiquement par `evaluate.py`. Les scores ci-dessous sont directement reutilisables dans la section Experimentation et evaluation du rapport technique.

## 1. Pertinence de la recherche vectorielle
- Precision top-1 : **71%** (5/7)
- MRR (top-3) : **0.786**

## 2. Robustesse du routage conversationnel
- Score global : **100%** (16/16 assertions)

Detail par scenario :
- [OK] Chaine d'ellipses sur les centres de sante (villes successives)
- [OK] Bascule volontaire centre -> pharmacie puis retour ville
- [OK] Question independante sur un theme de sante (ne doit heriter d'aucun sujet)
- [OK] Ambiguite lexicale pharmacie / etablissement
- [OK] Mention d'un etablissement sans signal de garde (reste centre)
- [OK] Ville seule en premiere question (ne doit jamais declencher la pharmacie par defaut)
- [OK] Faux positif lexical 'garde' dans une reponse assistant (ne doit pas changer le sujet)

## 3. Robustesse de la detection de ville (informatif)
- Taux de detection avec fautes de frappe incluses : **75%** (3/4)
- Limite connue : correspondance exacte, non tolerante aux fautes de frappe.

## 4. Taux de refus correct (anti-hallucination)
- Score : **100%** (6/6)

## 5. Routage bout en bout (generate_answer)
- Score : **100%** (2/2)

### Temps de reponse mesures
| Question | Mode | Recherche (ms) | Generation (ms) |
|---|---|---|---|
| Quels sont les symptomes du paludisme chez l'enfan... | recherche_vectorielle | 24 | 11315 |
| Quelle pharmacie est de garde a Ouagadougou ce soi... | scraping_live | 899 | 12205 |

## 6. Scenarios de non-regression (bugs reellement rencontres)
- Score global : **91%** (10/11 assertions)

Detail par scenario :
- [OK] Une salutation ne doit pas polluer la recherche (bug 'Salut, comment tu vas ?? symptomes du palu ?')
- [OK] Aucun nom de medicament ne doit apparaitre (regle 4 du prompt)
- [OK] Aucun nom de fichier interne ne doit fuiter (regle 10 du prompt)
- [OK] Aucune distance en km inventee pour une pharmacie (bug de la distance calculee cote serveur source)
- [OK] Le disclaimer d'urgence doit apparaitre face a un vrai symptome grave
- [ECHEC] Le disclaimer d'urgence ne doit PAS apparaitre pour une question factuelle (bug du rappel systematique)
- [OK] Nutrition apres dengue : pas de contamination de sujet (bug de sur-enrichissement de la requete)
- [OK] Question a sujets multiples : couverture des deux themes sans dilution
- [OK] Les sources ne doivent jamais etre dupliquees dans l'affichage

## 7. Variete des propositions de pharmacie
- Resultat : **ECHEC**
- Numeros proposes au 1er tour : aucun
- Numeros proposes au 2e tour : 25306168

---
*Rapport genere automatiquement -- a joindre en annexe du rapport technique.*