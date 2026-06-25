"""Fonctions utilitaires pour les chemins et l'affichage."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DOCS_DIR = ROOT_DIR / "docs"

DATA_PATH = DATA_DIR / "insurance_claims.csv"
MODEL_PATH = OUTPUTS_DIR / "fraud_detection_model.joblib"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"


def ensure_directories() -> None:
    """Crée les dossiers nécessaires si absents."""
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
