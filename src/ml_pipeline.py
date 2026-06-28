"""
Pipeline Machine Learning principal.

Ce fichier regroupe clairement les étapes demandées par l'encadrant :

1. Exploration de données
2. Transformation et nettoyage
3. Feature importance / variables influentes
4. Lancement de l'entraînement avec séparation X / y

Important :
- X contient les variables explicatives.
- y contient uniquement la target fraud_reported.
- fraud_reported ne doit jamais rester dans X.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "fraud_reported"


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage de base du dataset.

    - Remplace les valeurs '?' par NaN.
    - Supprime les lignes où la target est manquante.
    """
    cleaned = df.copy()
    cleaned = cleaned.replace("?", np.nan)

    if TARGET_COL in cleaned.columns:
        cleaned = cleaned.dropna(subset=[TARGET_COL])

    return cleaned


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sépare le dataset en X et y.

    X = toutes les colonnes utilisées pour prédire la fraude.
    y = la target fraud_reported.

    On isole fraud_reported dans y pour ne pas donner la réponse au modèle.
    """
    if target_col not in df.columns:
        raise ValueError(f"La colonne target '{target_col}' est introuvable dans le dataset.")

    # Colonnes supprimées car elles sont trop identifiantes ou inutiles pour l'entraînement.
    columns_to_drop = [col for col in ["policy_number", "insured_zip", "_c39"] if col in df.columns]

    X = df.drop(columns=[target_col] + columns_to_drop)
    y = df[target_col].map({"Y": 1, "N": 0})

    if y.isna().any():
        raise ValueError("La target contient des valeurs différentes de Y/N.")

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construit le préprocesseur qui transforme toutes les variables en numérique.

    - Variables numériques : imputation médiane + standardisation.
    - Variables catégorielles : imputation + OneHotEncoder.
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def prepare_train_test_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prépare les données pour l'entraînement.

    Étapes :
    1. Nettoyage
    2. Séparation X / y
    3. Split train/test avec stratification
    4. Préprocesseur pour transformer X en matrice numérique
    """
    cleaned = clean_raw_data(df)
    X, y = split_features_target(cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_test, y_train, y_test, preprocessor


def get_numeric_matrix_preview(X_train: pd.DataFrame, preprocessor: ColumnTransformer):
    """
    Transforme X_train en matrice numérique.

    Cette étape prouve que les modèles ML reçoivent uniquement des nombres.
    """
    X_train_numeric = preprocessor.fit_transform(X_train)
    return X_train_numeric


def load_feature_importance(feature_importance_path: Path) -> pd.DataFrame:
    """
    Charge les variables influentes générées après l'entraînement.
    """
    if not feature_importance_path.exists():
        raise FileNotFoundError(
            f"Le fichier {feature_importance_path} est introuvable. "
            "Lancez d'abord python train_model.py"
        )

    return pd.read_csv(feature_importance_path)
