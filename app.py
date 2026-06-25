"""Dashboard Streamlit pour la détection de fraude dans les sinistres d'assurance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.json"
FEATURE_IMPORTANCE_PATH = OUTPUTS_DIR / "feature_importance.csv"

st.set_page_config(
    page_title="Fraude Assurance ML",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data
def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if not FEATURE_IMPORTANCE_PATH.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(FEATURE_IMPORTANCE_PATH)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


st.title("🛡️ Détection de fraude dans les sinistres d’assurance")
st.markdown(
    """
Ce dashboard présente un système de Machine Learning qui analyse les dossiers de sinistres,
calcule une probabilité de fraude et classe chaque dossier par niveau de risque afin d'aider
les gestionnaires à prioriser les contrôles.

**Problématique :** Comment détecter automatiquement les dossiers de sinistres suspects afin de réduire le risque de fraude ?  
**Solution :** Modèle ML + scoring de risque + dashboard interactif pour visualiser les dossiers les plus suspects.
"""
)

if not PREDICTIONS_PATH.exists():
    st.error("Les prédictions sont absentes. Lance d'abord : `python train_model.py`")
    st.stop()

predictions = load_predictions()
metrics = load_metrics()
feature_importance = load_feature_importance()

# Sidebar filters
st.sidebar.header("🔎 Filtres")
risk_order = ["Faible", "Moyen", "Élevé"]
available_risks = [risk for risk in risk_order if risk in predictions["risk_level"].unique()]
selected_risks = st.sidebar.multiselect(
    "Niveau de risque",
    options=available_risks,
    default=available_risks,
)

min_prob, max_prob = float(predictions["fraud_probability"].min()), float(predictions["fraud_probability"].max())
prob_range = st.sidebar.slider(
    "Probabilité de fraude",
    min_value=0.0,
    max_value=1.0,
    value=(max(0.0, min_prob), min(1.0, max_prob)),
    step=0.01,
)

filtered = predictions[
    predictions["risk_level"].isin(selected_risks)
    & predictions["fraud_probability"].between(prob_range[0], prob_range[1])
].copy()

if "incident_type" in predictions.columns:
    incident_types = sorted(predictions["incident_type"].dropna().unique().tolist())
    selected_incidents = st.sidebar.multiselect(
        "Type de sinistre",
        options=incident_types,
        default=incident_types,
    )
    filtered = filtered[filtered["incident_type"].isin(selected_incidents)]

if "total_claim_amount" in predictions.columns:
    min_claim = int(predictions["total_claim_amount"].min())
    max_claim = int(predictions["total_claim_amount"].max())
    claim_range = st.sidebar.slider(
        "Montant total de réclamation",
        min_value=min_claim,
        max_value=max_claim,
        value=(min_claim, max_claim),
        step=1000,
    )
    filtered = filtered[filtered["total_claim_amount"].between(claim_range[0], claim_range[1])]

# KPIs
st.subheader("📌 Indicateurs clés")
col1, col2, col3, col4, col5 = st.columns(5)
total_dossiers = len(filtered)
suspected = int((filtered["prediction"] == 1).sum()) if total_dossiers else 0
high_risk = int((filtered["risk_level"] == "Élevé").sum()) if total_dossiers else 0
estimated_rate = suspected / total_dossiers if total_dossiers else 0
avg_probability = float(filtered["fraud_probability"].mean()) if total_dossiers else 0

col1.metric("Dossiers affichés", f"{total_dossiers:,}".replace(",", " "))
col2.metric("Fraudes suspectées", f"{suspected:,}".replace(",", " "))
col3.metric("Taux estimé", format_percent(estimated_rate))
col4.metric("Probabilité moyenne", format_percent(avg_probability))
col5.metric("Risque élevé", f"{high_risk:,}".replace(",", " "))

# Model performance
with st.expander("📊 Performance du modèle", expanded=True):
    best_model = metrics.get("best_model", "Non disponible")
    st.markdown(f"**Meilleur modèle sélectionné :** `{best_model}`")
    models_metrics = metrics.get("models", {})
    if models_metrics:
        metrics_table = pd.DataFrame(models_metrics).T.reset_index().rename(columns={"index": "model"})
        display_cols = ["model", "accuracy", "precision_fraud", "recall_fraud", "f1_fraud", "roc_auc"]
        st.dataframe(metrics_table[display_cols], use_container_width=True, hide_index=True)
        st.caption("Dans ce sujet, le recall fraude est important car il mesure la capacité à détecter les dossiers frauduleux.")

# Charts
st.subheader("📈 Visualisations")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    risk_counts = filtered["risk_level"].value_counts().reindex(risk_order).dropna().reset_index()
    risk_counts.columns = ["risk_level", "count"]
    fig_risk = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        title="Répartition par niveau de risque",
        text="count",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

with chart_col2:
    pred_counts = filtered["prediction_label"].value_counts().reset_index()
    pred_counts.columns = ["prediction_label", "count"]
    fig_pred = px.pie(
        pred_counts,
        names="prediction_label",
        values="count",
        title="Répartition fraude suspectée / non fraude",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_prob = px.histogram(
        filtered,
        x="fraud_probability",
        nbins=30,
        title="Distribution des probabilités de fraude",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with chart_col4:
    top10 = filtered.sort_values("fraud_probability", ascending=False).head(10).copy()
    top10["dossier"] = top10.get("policy_number", top10.index).astype(str)
    fig_top = px.bar(
        top10.sort_values("fraud_probability"),
        x="fraud_probability",
        y="dossier",
        orientation="h",
        title="Top 10 dossiers les plus suspects",
    )
    st.plotly_chart(fig_top, use_container_width=True)

if not feature_importance.empty:
    st.subheader("🧠 Variables les plus importantes")
    top_features = feature_importance.head(20).sort_values("importance")
    fig_features = px.bar(
        top_features,
        x="importance",
        y="feature",
        orientation="h",
        title="Top variables utilisées par le modèle",
    )
    st.plotly_chart(fig_features, use_container_width=True)

# Suspicious files table
st.subheader("📂 Liste des dossiers avec scoring")
priority_columns = [
    "policy_number",
    "fraud_reported",
    "fraud_probability",
    "prediction_label",
    "risk_level",
    "recommended_action",
    "incident_type",
    "incident_severity",
    "total_claim_amount",
    "insured_occupation",
    "insured_hobbies",
    "auto_make",
    "auto_model",
]
visible_columns = [col for col in priority_columns if col in filtered.columns]
if not visible_columns:
    visible_columns = filtered.columns.tolist()

table = filtered.sort_values("fraud_probability", ascending=False)[visible_columns].copy()
st.dataframe(table, use_container_width=True, hide_index=True)

csv_data = filtered.sort_values("fraud_probability", ascending=False).to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Télécharger les résultats filtrés en CSV",
    data=csv_data,
    file_name="fraud_predictions_filtered.csv",
    mime="text/csv",
)

st.info(
    "Ce système est un outil d’aide à la décision. La décision finale doit rester entre les mains d’un gestionnaire ou expert métier."
)
