"""
live_scraper.py
Scraping en temps réel des pharmacies de garde depuis infossante.net.

Ce module est appelé "à la demande" (pas au moment de l'ingestion), pour que
l'agent renvoie toujours la liste du JOUR, et pas une liste figée dans ChromaDB.

Source : https://infossante.net (structure HTML observée en juillet 2026 :
un tableau avec colonnes N°, Pharmacie, Emplacement, Contacts).
Si le site change de structure, seule la fonction parse_pharmacy_table()
est à adapter.
"""

import re
import requests
from bs4 import BeautifulSoup

URLS_PAR_VILLE = {
    "ouagadougou": "https://infossante.net/pharmacie-de-garde-de-ouagadougou",
    "bobo-dioulasso": "https://infossante.net/pharmacie-de-garde-de-bobo-dioulasso",
    "koudougou": "https://infossante.net/pharmacie-de-garde-de-koudougou",
    "ouahigouya": "https://infossante.net/pharmacie-de-garde-de-ouahigouya",
    "fada n'gourma": "https://infossante.net/pharmacie-de-garde-de-fada-ngourma",
}

# Alias courts / variantes courantes utilisées dans le langage parlé
VILLE_ALIASES = {
    "ouagadougou": ["ouaga"],
    "bobo-dioulasso": ["bobo", "bobo dioulasso"],
    "koudougou": [],
    "ouahigouya": [],
    "fada n'gourma": ["fada"],
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}


def detect_city(query: str) -> str:
    """Détecte la ville mentionnée dans la question (nom complet ou alias). Ouagadougou par défaut."""
    query_lower = query.lower()
    for ville, aliases in VILLE_ALIASES.items():
        candidats = [ville, ville.replace("-", " ")] + aliases
        if any(c in query_lower for c in candidats):
            return ville
    return "ouagadougou"


def get_pharmacies_de_garde(query: str = "", max_results: int = 15) -> dict:
    """
    Scrape la page infossante.net correspondant à la ville détectée dans la requête
    et renvoie une liste de pharmacies de garde du jour.
    """
    ville = detect_city(query)
    url = URLS_PAR_VILLE.get(ville, URLS_PAR_VILLE["ouagadougou"])

    response = None
    last_error = None
    for attempt in range(2):  # 1 essai + 1 nouvel essai en cas de raté ponctuel
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            break
        except requests.RequestException as e:
            last_error = e
            response = None

    if response is None:
        return {
            "success": False,
            "ville": ville,
            "error": str(last_error),
            "pharmacies": [],
        }

    soup = BeautifulSoup(response.text, "html.parser")

    # Le HTML de cette page contient une erreur de structure : à partir de la 2e ligne,
    # les balises <tr> ne sont plus correctement imbriquées dans le <table> (probablement
    # une balise mal fermée côté site). Le navigateur corrige ça automatiquement, mais pas
    # notre parser. Solution : chercher TOUS les <tr> de la page entière plutôt que
    # seulement ceux à l'intérieur du <table> détecté, pour ne rater aucune pharmacie.
    all_rows = soup.find_all("tr")

    if not all_rows:
        return {
            "success": False,
            "ville": ville,
            "error": "Structure de page inattendue (aucune ligne trouvée).",
            "pharmacies": [],
        }

    pharmacies = []
    for row in all_rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # La ligne d'en-tête a "N°" au lieu d'un numéro : on l'ignore.
        first_cell_text = cells[0].get_text(strip=True)
        if not first_cell_text.isdigit():
            continue

        nom = cells[1].get_text(strip=True)
        emplacement = cells[2].get_text(strip=True)

        # Le site source calcule une distance côté navigateur (géolocalisation du visiteur
        # qui a chargé la page au moment du scraping) — elle n'a AUCUN rapport avec la
        # position réelle de l'utilisateur final. On la retire pour ne pas induire en erreur.
        emplacement = re.sub(r"\s*situé[e]?\s*à\s*[\d.,]+\s*Km\s*de\s*vous", "", emplacement, flags=re.IGNORECASE).strip()

        # Le numéro de téléphone et le lien Google Maps sont dans la même cellule (avec
        # aussi le texte "En garde"). On extrait chacun proprement via les balises <a>.
        contact_cell = cells[3]
        phone = ""
        itineraire_url = ""
        tel_link = contact_cell.find("a", href=lambda h: h and h.startswith("tel:"))
        if tel_link:
            phone = tel_link.get_text(strip=True)

        # Ce lien Google Maps ne contient QUE les coordonnées de la pharmacie (daddr=lat,lng),
        # pas de position de départ : Google Maps calculera le trajet à partir de la position
        # réelle de la personne qui ouvre le lien sur son propre appareil. Fiable, contrairement
        # à la distance en km calculée côté serveur du site source.
        maps_link = contact_cell.find("a", href=lambda h: h and "maps.google.com" in h)
        if maps_link and maps_link.get("href"):
            itineraire_url = maps_link["href"]

        pharmacies.append({
            "nom": nom,
            "emplacement": emplacement,
            "contact": phone,
            "itineraire_url": itineraire_url,
        })

        if len(pharmacies) >= max_results:
            break

    return {
        "success": True,
        "ville": ville,
        "source_url": url,
        "pharmacies": pharmacies,
    }


def format_for_llm(result: dict) -> str:
    """Formate le résultat du scraping en texte lisible à injecter dans le prompt du LLM."""
    if not result["success"] or not result["pharmacies"]:
        return (
            f"Impossible de récupérer les pharmacies de garde en direct pour "
            f"{result['ville'].title()} actuellement (source indisponible)."
        )

    lines = [f"Pharmacies de garde aujourd'hui à {result['ville'].title()} (source : infossante.net) :"]
    for p in result["pharmacies"]:
        line = f"- {p['nom']} — {p['emplacement']}"
        if p["contact"]:
            line += f" — Contact : {p['contact']}"
        if p.get("itineraire_url"):
            line += f" — Itinéraire : {p['itineraire_url']}"
        lines.append(line)

    return "\n".join(lines)


if __name__ == "__main__":
    # Test rapide en ligne de commande
    result = get_pharmacies_de_garde("pharmacie de garde à Ouagadougou")
    print(format_for_llm(result))