"""
Script de démonstration du pipeline Machine Learning.

Commande :
    py run_ml_pipeline.py

Ce script montre dans le terminal :
1. Exploration des données
2. Nettoyage et transformation
3. Séparation X / y
4. Transformation de X en matrice numérique
5. Variables influentes si le fichier existe
"""

from __future__ import annotations

from pathlib import Path

from src.data_exploration import load_dataset, print_exploration_report
from src.ml_pipeline import (
    prepare_train_test_data,
    get_numeric_matrix_preview,
    load_feature_importance,
)

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "insurance_claims.csv"
FEATURE_IMPORTANCE_PATH = ROOT_DIR / "outputs" / "feature_importance.csv"


def main() -> None:
    df = load_dataset(DATA_PATH)

    # 1. Exploration
    print_exploration_report(df)

    # 2, 3, 4. Nettoyage, séparation X/y, transformation numérique
    print("\n" + "=" * 70)
    print("2. TRANSFORMATION / NETTOYAGE + 3. SÉPARATION X / y")
    print("=" * 70)

    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(df)

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_test shape  : {y_test.shape}")

    print("\nVérification importante : la colonne fraud_reported n'est pas dans X.")
    print("fraud_reported dans X_train ?", "fraud_reported" in X_train.columns)

    print("\nTransformation de X_train en matrice numérique...")
    X_train_numeric = get_numeric_matrix_preview(X_train, preprocessor)
    print(f"Matrice numérique obtenue : {X_train_numeric.shape}")
    print("Le modèle reçoit donc une matrice numérique sans la target.")

    # 5. Variables influentes
    print("\n" + "=" * 70)
    print("4. VARIABLES INFLUENTES")
    print("=" * 70)

    if FEATURE_IMPORTANCE_PATH.exists():
        feature_importance = load_feature_importance(FEATURE_IMPORTANCE_PATH)
        print(feature_importance.head(10).to_string(index=False))
    else:
        print("Fichier feature_importance.csv introuvable.")
        print("Lancez d'abord : py train_model.py")

    print("\nPipeline terminé.")


if __name__ == "__main__":
    main()
