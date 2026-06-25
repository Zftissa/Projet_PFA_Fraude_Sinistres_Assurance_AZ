"""Script principal pour entraîner le modèle de détection de fraude."""

from __future__ import annotations

import pandas as pd

from src.model_training import (
    extract_feature_importance,
    save_json,
    save_model_bundle,
    train_and_compare_models,
)
from src.preprocessing import load_data, prepare_features_target, split_data
from src.risk_scoring import add_risk_scoring
from src.utils import (
    DATA_PATH,
    FEATURE_IMPORTANCE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    ensure_directories,
)


def main() -> None:
    ensure_directories()

    print("Chargement du dataset...")
    df = load_data(DATA_PATH)
    print(f"Dataset chargé : {df.shape[0]} lignes et {df.shape[1]} colonnes")

    print("Préparation des features et de la cible...")
    X, y = prepare_features_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Entraînement et comparaison des modèles...")
    best_model_name, best_pipeline, all_metrics = train_and_compare_models(X_train, X_test, y_train, y_test)

    print(f"Meilleur modèle sélectionné : {best_model_name}")
    print("Ré-entraînement du meilleur modèle sur toutes les données pour le dashboard...")
    best_pipeline.fit(X, y)

    probabilities = best_pipeline.predict_proba(X)[:, 1]
    predictions = add_risk_scoring(df, probabilities)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    feature_importance = extract_feature_importance(best_pipeline)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    metrics_output = {
        "dataset": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "fraud_count_Y": int((df["fraud_reported"] == "Y").sum()),
            "non_fraud_count_N": int((df["fraud_reported"] == "N").sum()),
            "fraud_rate": round(float((df["fraud_reported"] == "Y").mean()), 4),
        },
        "best_model": best_model_name,
        "models": all_metrics,
        "selection_rule": "Meilleur modèle choisi selon F1-score fraude, puis recall fraude, puis ROC-AUC.",
        "risk_thresholds": {
            "Faible": "fraud_probability < 0.40",
            "Moyen": "0.40 <= fraud_probability < 0.70",
            "Élevé": "fraud_probability >= 0.70",
        },
    }
    save_json(metrics_output, METRICS_PATH)
    save_model_bundle(best_pipeline, MODEL_PATH, best_model_name, X.columns.tolist(), metrics_output)

    print("\n=== Résumé ===")
    print(f"Meilleur modèle : {best_model_name}")
    for name, metric in all_metrics.items():
        print(
            f"- {name}: accuracy={metric['accuracy']} | "
            f"precision_fraude={metric['precision_fraud']} | "
            f"recall_fraude={metric['recall_fraud']} | "
            f"f1_fraude={metric['f1_fraud']} | roc_auc={metric['roc_auc']}"
        )

    print(f"\nModèle sauvegardé : {MODEL_PATH}")
    print(f"Métriques sauvegardées : {METRICS_PATH}")
    print(f"Prédictions sauvegardées : {PREDICTIONS_PATH}")
    print(f"Importance variables sauvegardée : {FEATURE_IMPORTANCE_PATH}")


if __name__ == "__main__":
    main()
