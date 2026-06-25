"""Prétraitement des données pour le projet fraude assurance."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "fraud_reported"
COLUMNS_TO_DROP = [
    "policy_number",      # identifiant, pas utile pour apprendre un comportement
    "insured_zip",        # information très spécifique / sensible
    "incident_location",  # adresse textuelle très détaillée
    "_c39",               # colonne vide dans le dataset
    "policy_bind_date",   # remplacée par des variables dérivées
    "incident_date",      # remplacée par des variables dérivées
]


def load_data(path: str | Path) -> pd.DataFrame:
    """Charge le fichier CSV et harmonise les valeurs manquantes."""
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)
    return df


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crée des variables utiles à partir des dates de police et de sinistre."""
    data = df.copy()

    if "policy_bind_date" in data.columns:
        data["policy_bind_date"] = pd.to_datetime(data["policy_bind_date"], errors="coerce")
    if "incident_date" in data.columns:
        data["incident_date"] = pd.to_datetime(data["incident_date"], errors="coerce")

    if "policy_bind_date" in data.columns and "incident_date" in data.columns:
        data["policy_age_days"] = (data["incident_date"] - data["policy_bind_date"]).dt.days
        data["incident_month"] = data["incident_date"].dt.month
        data["incident_dayofweek"] = data["incident_date"].dt.dayofweek
        data["policy_bind_year"] = data["policy_bind_date"].dt.year

    return data


def prepare_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prépare X et y à partir du dataset brut."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"La colonne cible '{TARGET_COLUMN}' est absente du dataset.")

    data = add_date_features(df)

    y = data[TARGET_COLUMN].map({"Y": 1, "N": 0})
    if y.isna().any():
        raise ValueError("La cible doit contenir seulement les valeurs 'Y' et 'N'.")

    columns_to_drop = [col for col in COLUMNS_TO_DROP + [TARGET_COLUMN] if col in data.columns]
    X = data.drop(columns=columns_to_drop)
    return X, y.astype(int)


def get_column_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Retourne les colonnes numériques et catégorielles."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Construit le préprocesseur sklearn pour les variables numériques et catégorielles."""
    numeric_features, categorical_features = get_column_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # compatibilité anciennes versions scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.20, random_state: int = 42):
    """Sépare les données avec stratification pour garder la même proportion fraude/non fraude."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
