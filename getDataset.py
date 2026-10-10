import csv
import time
import os
import requests

from pathlib import Path
from collections import defaultdict, deque

# Competition a recuperer
COMPETITION = "PL"  # Premier League

# Annees de debut des saisons
SEASONS = [
    2020, 2021, 2022, 2023, 2024,
    2025, 2019, 2018, 2017, 2016
]

# Nombre de matchs precedents utilises
LAST_N_MATCHES = 5

# Dossier des fichiers CSV
DATA_DIR = Path(__file__).resolve().parent / "dataset"
CSV_FILE = DATA_DIR / "dataset.csv"
RESULT_FILE = DATA_DIR / "result.csv"

# Cle API : configure FOOTBALL_DATA_TOKEN dans ton environnement
# Ne publie jamais ta cle sur GitHub
API_TOKEN = "PUT YOUR API KEY HERE"


HEADERS = {
    "X-Auth-Token": API_TOKEN
}

BASE_URL = (
    f"https://api.football-data.org/v4/"
    f"competitions/{COMPETITION}/matches"
)


def get_matches():
    """Recupere les matchs termines pour chaque saison."""

    all_matches = []

    for season in SEASONS:
        print(f"Recuperation de la saison {season}...")

        try:
            response = requests.get(
                BASE_URL,
                headers=HEADERS,
                params={
                    "season": season,
                    "status": "FINISHED"
                },
                timeout=30
            )

        except requests.RequestException as error:
            print(f"Erreur reseau pour {season} : {error}")
            continue

        if response.status_code == 401:
            raise RuntimeError(
                "Cle API invalide. Verifie ta nouvelle cle."
            )

        if response.status_code == 403:
            print(f"Saison {season} inaccessible, ignoree.")
            continue

        if response.status_code == 429:
            raise RuntimeError(
                "Limite de requetes atteinte. "
                "Attends avant de relancer."
            )

        response.raise_for_status()

        season_matches = response.json().get("matches", [])
        all_matches.extend(season_matches)

        print(f"{len(season_matches)} matchs recuperes.")

        time.sleep(1)

    # Supprime les doublons eventuels
    unique_matches = {
        match["id"]: match
        for match in all_matches
    }

    # Trie tous les matchs par date
    return sorted(
        unique_matches.values(),
        key=lambda match: match["utcDate"]
    )


def new_history():
    """Conserve les 5 derniers matchs d'une equipe."""

    return deque(maxlen=LAST_N_MATCHES)


def calculate_stats(history):
    """Calcule les statistiques des matchs precedents."""

    if not history:
        return {
            "goals_scored": 0.0,
            "goals_conceded": 0.0,
            "points": 0.0
        }

    # Moyenne de buts par match
    average_goals_scored = (
        sum(game["gf"] for game in history) / len(history)
    )

    average_goals_conceded = (
        sum(game["ga"] for game in history) / len(history)
    )

    # Total des points sur les 5 derniers matchs
    total_points = sum(
        game["points"] for game in history
    )

    return {
        "goals_scored": average_goals_scored,
        "goals_conceded": average_goals_conceded,
        "points": total_points
    }


def add_result(history, goals_for, goals_against):
    """Ajoute un match a l'historique d'une equipe."""

    if goals_for > goals_against:
        points = 3
    elif goals_for == goals_against:
        points = 1
    else:
        points = 0

    history.append({
        "gf": goals_for,
        "ga": goals_against,
        "points": points
    })


def normalize_goals(average_goals):
    """Normalise une moyenne de buts en la divisant par 5."""

    return average_goals / 5.0


def normalize_points(total_points):
    """Normalise les points sur 5 matchs entre 0 et 1."""

    return total_points / 15.0


def create_dataset(matches):
    """Cree les statistiques normalisees et les resultats."""

    team_history = defaultdict(new_history)

    dataset_rows = []
    result_rows = []

    for match in matches:

        if match.get("status") != "FINISHED":
            continue

        score = match.get("score", {}).get("fullTime", {})

        home_goals = score.get("home")
        away_goals = score.get("away")

        if home_goals is None or away_goals is None:
            continue

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]

        # Recuperation de l'historique AVANT le match actuel
        home_history = team_history[home_team]
        away_history = team_history[away_team]

        # On attend que les deux equipes aient 5 matchs precedents
        if (
            len(home_history) == LAST_N_MATCHES
            and len(away_history) == LAST_N_MATCHES
        ):
            # Statistiques de l'equipe a domicile
            home = calculate_stats(home_history)

            # Statistiques de l'equipe a l'exterieur
            away = calculate_stats(away_history)

            # Normalisation des 6 entrees
            dataset_rows.append({
                "home_goals_scored": normalize_goals(
                    home["goals_scored"]
                ),

                "home_goals_conceded": normalize_goals(
                    home["goals_conceded"]
                ),

                "home_points": normalize_points(
                    home["points"]
                ),

                "away_goals_scored": normalize_goals(
                    away["goals_scored"]
                ),

                "away_goals_conceded": normalize_goals(
                    away["goals_conceded"]
                ),

                "away_points": normalize_points(
                    away["points"]
                )
            })

            # 1 si l'equipe a domicile gagne
            # 0 si elle perd ou fait match nul
            home_result = int(home_goals > away_goals)

            # Sauvegarde le resultat dans result.csv
            result_rows.append({
                "result": home_result
            })

        # Met a jour l'historique APRES avoir traite le match
        add_result(
            home_history,
            home_goals,
            away_goals
        )

        add_result(
            away_history,
            away_goals,
            home_goals
        )

    return dataset_rows, result_rows


def save_csv(rows, file_path, fieldnames):
    """Enregistre les donnees dans un fichier CSV."""

    try:
        with open(
            file_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    except PermissionError:
        print(
            f"Impossible d'ecrire {file_path.name}. "
            "Ferme le fichier s'il est ouvert dans Excel."
        )
        return False

    print(
        f"{file_path.name} sauvegarde : "
        f"{len(rows)} lignes de donnees."
    )

    return True


def main():
    # Cree le dossier dataset s'il n'existe pas
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Recupere les matchs depuis l'API
    matches = get_matches()

    print(
        f"\nTotal de matchs recuperes : {len(matches)}"
    )

    if not matches:
        print(
            "Aucun match recupere. "
            "Verifie ta cle API et les saisons."
        )
        return

    # Cree les statistiques et les resultats
    dataset_rows, result_rows = create_dataset(matches)

    # Sauvegarde les deux fichiers separement
    dataset_saved = save_csv(
        dataset_rows,
        CSV_FILE,
        [
            "home_goals_scored",
            "home_goals_conceded",
            "home_points",
            "away_goals_scored",
            "away_goals_conceded",
            "away_points"
        ]
    )

    result_saved = save_csv(
        result_rows,
        RESULT_FILE,
        ["result"]
    )

    # Verifie que les deux fichiers sont alignes
    print("\nTermine.")
    print(f"Entrees : {len(dataset_rows)}")
    print(f"Resultats : {len(result_rows)}")

    if len(dataset_rows) != len(result_rows):
        print("ERREUR : les fichiers ne sont pas alignes.")

    elif dataset_saved and result_saved:
        print("Les deux fichiers sont alignes et sauvegardes.")


if __name__ == "__main__":
    main()