from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"

RISK_ORDER = ["Faible", "Moyen", "Élevé"]
RISK_COLORS = {
    "Faible": "#2ECC71",
    "Moyen": "#F39C12",
    "Élevé": "#E74C3C",
}
PREDICTION_COLORS = {
    "Non fraude suspectée": "#3498DB",
    "Fraude suspectée": "#E74C3C",
}

DISPLAY_NAMES = {
    "policy_number": "N° dossier / police",
    "fraud_probability": "Probabilité fraude",
    "dashboard_prediction_label": "Décision",
    "risk_level": "Niveau de risque",
    "recommended_action": "Action recommandée",
    "incident_date": "Date sinistre",
    "incident_type": "Type sinistre",
    "incident_city": "Ville sinistre",
    "incident_state": "État sinistre",
    "incident_severity": "Gravité",
    "collision_type": "Type collision",
    "total_claim_amount": "Montant total",
    "injury_claim": "Montant blessures",
    "property_claim": "Montant biens",
    "vehicle_claim": "Montant véhicule",
    "authorities_contacted": "Autorité contactée",
    "police_report_available": "Rapport police",
    "insured_occupation": "Profession assuré",
    "insured_hobbies": "Loisir assuré",
    "auto_make": "Marque auto",
    "auto_model": "Modèle auto",
    "auto_year": "Année auto",
}

st.set_page_config(
    page_title="Fraude Assurance ML",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    return pd.read_csv(PREDICTIONS_PATH)


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def add_dynamic_decision_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = df.copy()
    output["dashboard_prediction"] = (output["fraud_probability"] >= threshold).astype(int)
    output["dashboard_prediction_label"] = output["dashboard_prediction"].map(
        {0: "Non fraude suspectée", 1: "Fraude suspectée"}
    )
    return output


def prepare_display_table(df: pd.DataFrame, columns: list[str], percent: bool = True) -> pd.DataFrame:
    visible_columns = [col for col in columns if col in df.columns]
    table = df[visible_columns].copy()
    if percent and "fraud_probability" in table.columns:
        table["fraud_probability"] = (table["fraud_probability"] * 100).round(1).astype(str) + "%"
    return table.rename(columns={col: DISPLAY_NAMES.get(col, col) for col in table.columns})


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    priority_columns = [
        "policy_number",
        "fraud_probability",
        "dashboard_prediction_label",
        "risk_level",
        "recommended_action",
        "incident_date",
        "incident_type",
        "incident_city",
        "incident_severity",
        "collision_type",
        "total_claim_amount",
        "injury_claim",
        "property_claim",
        "vehicle_claim",
        "authorities_contacted",
        "police_report_available",
        "insured_occupation",
        "insured_hobbies",
        "auto_make",
        "auto_model",
        "auto_year",
    ]
    visible_columns = [col for col in priority_columns if col in df.columns]
    if not visible_columns:
        visible_columns = [col for col in df.columns if col != "fraud_reported"]
    sorted_df = df.sort_values("fraud_probability", ascending=False)
    return prepare_display_table(sorted_df, visible_columns, percent=True)


st.title("🛡️ Détection de fraude dans les sinistres d’assurance")

if not PREDICTIONS_PATH.exists():
    st.error("Les prédictions sont absentes. Lance d'abord : `python train_model.py`")
    st.stop()

predictions_raw = load_predictions()

st.sidebar.header("⚙️ Paramètres")
decision_threshold = st.sidebar.slider(
    "Seuil de décision fraude",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01,
)

predictions = add_dynamic_decision_columns(predictions_raw, decision_threshold)

st.sidebar.header("🔎 Filtres")
available_risks = [risk for risk in RISK_ORDER if risk in predictions["risk_level"].unique()]
selected_risks = st.sidebar.multiselect(
    "Niveau de risque",
    options=available_risks,
    default=available_risks,
)

prob_range = st.sidebar.slider(
    "Probabilité de fraude",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
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

st.subheader("📌 Indicateurs clés")
col1, col2, col3, col4, col5 = st.columns(5)
total_dossiers = len(filtered)
suspected = int((filtered["dashboard_prediction"] == 1).sum()) if total_dossiers else 0
high_risk = int((filtered["risk_level"] == "Élevé").sum()) if total_dossiers else 0
estimated_rate = safe_rate(suspected, total_dossiers)
avg_probability = float(filtered["fraud_probability"].mean()) if total_dossiers else 0.0

col1.metric("Dossiers affichés", f"{total_dossiers:,}".replace(",", " "))
col2.metric("Fraudes suspectées", f"{suspected:,}".replace(",", " "))
col3.metric("Taux estimé", format_percent(estimated_rate))
col4.metric("Probabilité moyenne", format_percent(avg_probability))
col5.metric("Risque élevé", f"{high_risk:,}".replace(",", " "))

if filtered.empty:
    st.warning("Aucun dossier ne correspond aux filtres sélectionnés.")
    st.stop()

# Dashboard en une seule page
st.subheader("Vue générale des risques")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    risk_counts = (
        filtered["risk_level"]
        .value_counts()
        .reindex(RISK_ORDER)
        .dropna()
        .reset_index()
    )
    risk_counts.columns = ["risk_level", "count"]
    fig_risk = px.bar(
        risk_counts,
        x="risk_level",
        y="count",
        title="Répartition par niveau de risque",
        text="count",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
    )
    fig_risk.update_layout(showlegend=False)
    st.plotly_chart(fig_risk, use_container_width=True)

with chart_col2:
    pred_counts = filtered["dashboard_prediction_label"].value_counts().reset_index()
    pred_counts.columns = ["prediction_label", "count"]
    fig_pred = px.pie(
        pred_counts,
        names="prediction_label",
        values="count",
        title=f"Décision selon le seuil {decision_threshold:.2f}",
        color="prediction_label",
        color_discrete_map=PREDICTION_COLORS,
    )
    st.plotly_chart(fig_pred, use_container_width=True)

st.divider()
st.subheader("Analyse des probabilités de fraude")
chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    fig_prob = px.histogram(
        filtered,
        x="fraud_probability",
        nbins=30,
        title="Distribution des probabilités de fraude",
    )
    fig_prob.add_vline(
        x=decision_threshold,
        line_dash="dash",
        annotation_text="Seuil",
        annotation_position="top right",
    )
    st.plotly_chart(fig_prob, use_container_width=True)

with chart_col4:
    top10 = filtered.sort_values("fraud_probability", ascending=False).head(10).copy()
    top10["dossier"] = top10.get("policy_number", top10.index).astype(str)
    top10["dossier_label"] = "Dossier " + top10["dossier"]
    top10["probabilité fraude"] = top10["fraud_probability"] * 100
    top10_chart = top10.sort_values("fraud_probability", ascending=True).copy()
    fig_top = px.bar(
        top10_chart,
        x="probabilité fraude",
        y="dossier_label",
        orientation="h",
        title="Top 10 dossiers les plus suspects",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        text="probabilité fraude",
        labels={
            "probabilité fraude": "Probabilité de fraude (%)",
            "dossier_label": "N° dossier / police",
            "risk_level": "Niveau de risque",
        },
        hover_data={
            "dossier": True,
            "fraud_probability": ":.2%",
            "probabilité fraude": False,
            "risk_level": True,
        },
    )
    fig_top.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_top.update_layout(
        xaxis_range=[0, 105],
        yaxis_title="N° dossier / police",
        xaxis_title="Probabilité de fraude (%)",
        yaxis=dict(type="category", categoryorder="array", categoryarray=top10_chart["dossier_label"].tolist()),
    )
    st.plotly_chart(fig_top, use_container_width=True)

st.subheader("📋 Top 10 dossiers à rechercher")
top10_table_columns = [
    "policy_number",
    "fraud_probability",
    "risk_level",
    "recommended_action",
    "incident_date",
    "incident_type",
    "incident_city",
    "incident_severity",
    "total_claim_amount",
    "police_report_available",
]
st.dataframe(
    prepare_display_table(top10.sort_values("fraud_probability", ascending=False), top10_table_columns, percent=True),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("📂 Liste des dossiers à contrôler")
only_suspects = st.checkbox("Afficher seulement les dossiers suspectés", value=False)
cases_view = filtered.copy()
if only_suspects:
    cases_view = cases_view[cases_view["dashboard_prediction"] == 1]

table = build_main_table(cases_view)
st.dataframe(table, use_container_width=True, hide_index=True)

csv_data = table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Télécharger les dossiers filtrés en CSV",
    data=csv_data,
    file_name="fraud_predictions_filtered.csv",
    mime="text/csv",
)
