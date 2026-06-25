"""Entraînement, comparaison et sauvegarde des modèles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_preprocessor

RANDOM_STATE = 42


def get_candidate_models() -> Dict[str, object]:
    """Retourne les modèles à tester."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=7,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    """Évalue un modèle sur le test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_fraud": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall_fraud": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_fraud": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def train_and_compare_models(X_train, X_test, y_train, y_test) -> Tuple[str, Pipeline, Dict[str, object]]:
    """Entraîne plusieurs modèles et sélectionne le meilleur."""
    results: Dict[str, object] = {}
    trained_models: Dict[str, Pipeline] = {}

    for model_name, estimator in get_candidate_models().items():
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)
        results[model_name] = evaluate_model(pipeline, X_test, y_test)
        trained_models[model_name] = pipeline

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["f1_fraud"],
            results[name]["recall_fraud"],
            results[name]["roc_auc"],
        ),
    )
    return best_model_name, trained_models[best_model_name], results


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """Récupère les noms des variables après OneHotEncoding."""
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    numeric_features = preprocessor.transformers_[0][2]
    feature_names.extend(list(numeric_features))

    categorical_features = preprocessor.transformers_[1][2]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]
    encoder = categorical_pipeline.named_steps["encoder"]
    encoded_names = encoder.get_feature_names_out(categorical_features)
    feature_names.extend(encoded_names.tolist())

    return feature_names


def extract_feature_importance(pipeline: Pipeline, top_n: int = 40) -> pd.DataFrame:
    """Extrait l'importance des variables si disponible."""
    model = pipeline.named_steps["model"]
    feature_names = get_feature_names(pipeline)

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)
    return importance_df


def save_json(data: Dict[str, object], path: str | Path) -> None:
    """Sauvegarde un dictionnaire en JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def save_model_bundle(
    pipeline: Pipeline,
    path: str | Path,
    best_model_name: str,
    feature_columns: list[str],
    metrics: Dict[str, object],
) -> None:
    """Sauvegarde le pipeline et les métadonnées utiles."""
    bundle = {
        "pipeline": pipeline,
        "best_model_name": best_model_name,
        "feature_columns": feature_columns,
        "metrics": metrics,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
