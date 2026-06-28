"""
Module d'exploration des données.

Ce fichier répond à la première étape du pipeline demandé :
1. Exploration de données

Il permet de comprendre le dataset avant le nettoyage et l'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Charge le dataset CSV."""
    return pd.read_csv(data_path)


def explore_dataset(df: pd.DataFrame, target_col: str = "fraud_reported") -> dict:
    """
    Retourne un résumé de l'exploration du dataset :
    - taille du dataset
    - colonnes disponibles
    - types des variables
    - valeurs manquantes
    - valeurs inconnues représentées par '?'
    - répartition de la target
    """
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "unknown_question_mark": (df == "?").sum().to_dict(),
    }

    if target_col in df.columns:
        summary["target_distribution"] = df[target_col].value_counts().to_dict()
        summary["target_distribution_percent"] = (
            df[target_col].value_counts(normalize=True).mul(100).round(2).to_dict()
        )

    return summary


def print_exploration_report(df: pd.DataFrame, target_col: str = "fraud_reported") -> None:
    """Affiche un rapport simple d'exploration des données."""
    summary = explore_dataset(df, target_col=target_col)

    print("=" * 70)
    print("1. EXPLORATION DES DONNÉES")
    print("=" * 70)

    print(f"Taille du dataset : {summary['shape'][0]} lignes et {summary['shape'][1]} colonnes")

    print("\nColonnes disponibles :")
    for col in summary["columns"]:
        print(f"- {col}")

    if "target_distribution" in summary:
        print(f"\nRépartition de la variable cible '{target_col}' :")
        for label, count in summary["target_distribution"].items():
            percent = summary["target_distribution_percent"][label]
            print(f"- {label} : {count} dossiers ({percent}%)")

    missing = {k: v for k, v in summary["missing_values"].items() if v > 0}
    unknown = {k: v for k, v in summary["unknown_question_mark"].items() if v > 0}

    print("\nValeurs manquantes détectées :")
    print(missing if missing else "Aucune valeur manquante NaN détectée.")

    print("\nValeurs inconnues '?' détectées :")
    print(unknown if unknown else "Aucune valeur '?' détectée.")
