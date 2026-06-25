"""Fonctions de scoring métier pour classer les dossiers selon le risque."""

from __future__ import annotations

import pandas as pd


LOW_THRESHOLD = 0.40
HIGH_THRESHOLD = 0.70


def assign_risk_level(probability: float) -> str:
    """Retourne le niveau de risque à partir d'une probabilité de fraude."""
    if probability >= HIGH_THRESHOLD:
        return "Élevé"
    if probability >= LOW_THRESHOLD:
        return "Moyen"
    return "Faible"


def recommended_action(risk_level: str) -> str:
    """Retourne l'action métier recommandée selon le niveau de risque."""
    actions = {
        "Faible": "Traitement normal",
        "Moyen": "Vérification complémentaire",
        "Élevé": "Contrôle approfondi par gestionnaire",
    }
    return actions.get(risk_level, "À vérifier")


def add_risk_scoring(df: pd.DataFrame, probabilities, threshold: float = 0.50) -> pd.DataFrame:
    """Ajoute les colonnes de probabilité, prédiction, niveau de risque et action."""
    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = (result["fraud_probability"] >= threshold).astype(int)
    result["prediction_label"] = result["prediction"].map({1: "Fraude suspectée", 0: "Non fraude"})
    result["risk_level"] = result["fraud_probability"].apply(assign_risk_level)
    result["recommended_action"] = result["risk_level"].apply(recommended_action)
    return result
