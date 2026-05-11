import pandas as pd
import streamlit as st

from src.utils.database import get_client
from src.utils.monitoring_stats import (
    build_evolution_chart,
    build_histogram,
    compute_metrics,
)

st.title("Monitoring")

user = st.session_state.user

if user is None or user.get("role") != "administrateur":
    st.error("Accès refusé. Cette page est réservée aux administrateurs.")
    st.stop()

db = get_client()

# --- Chargement des données ---

# 1. Logs récents (50) pour les métriques
logs_50_resp = (
    db.table("prediction_logs")
    .select("sk_id_curr, inference_time_ms")
    .eq("found", True)
    .order("requested_at", desc=True)
    .limit(50)
    .execute()
)
df_logs_50 = pd.DataFrame(logs_50_resp.data) if logs_50_resp.data else pd.DataFrame()

# 2. Logs récents (500) pour l'évolution hebdo
logs_500_resp = (
    db.table("prediction_logs")
    .select("sk_id_curr, requested_at")
    .eq("found", True)
    .order("requested_at", desc=True)
    .limit(500)
    .execute()
)
df_logs_500 = pd.DataFrame(logs_500_resp.data) if logs_500_resp.data else pd.DataFrame()

# 3. Prédictions pour les sk_id consultés (50 derniers)
df_preds_50 = pd.DataFrame()
if not df_logs_50.empty:
    sk_ids_50 = df_logs_50["sk_id_curr"].tolist()
    preds_50_resp = (
        db.table("predictions")
        .select("sk_id_curr, proba_class_1")
        .in_("sk_id_curr", sk_ids_50)
        .execute()
    )
    df_preds_50 = pd.DataFrame(preds_50_resp.data) if preds_50_resp.data else pd.DataFrame()

# 4. Prédictions pour les sk_id consultés (500 derniers)
df_preds_500 = pd.DataFrame()
if not df_logs_500.empty:
    sk_ids_500 = df_logs_500["sk_id_curr"].unique().tolist()
    preds_500_resp = (
        db.table("predictions")
        .select("sk_id_curr, proba_class_1")
        .in_("sk_id_curr", sk_ids_500)
        .execute()
    )
    df_preds_500 = pd.DataFrame(preds_500_resp.data) if preds_500_resp.data else pd.DataFrame()

# 5. Toutes les prédictions (sample 5000) pour l'histogramme
hist_resp = (
    db.table("predictions")
    .select("proba_class_1")
    .limit(5000)
    .execute()
)
df_hist = pd.DataFrame(hist_resp.data) if hist_resp.data else pd.DataFrame()

# --- Section 1 : Métriques clés ---

st.subheader("Métriques clés")
st.caption("Calculées sur les 50 dernières requêtes abouties.")

if df_logs_50.empty or df_preds_50.empty:
    st.info("Pas encore assez de données pour afficher les métriques.")
else:
    metrics = compute_metrics(df_logs_50, df_preds_50)
    col1, col2, col3 = st.columns(3)
    col1.metric("Taux de défaut prédit", f"{metrics['taux_defaut']:.1f}%")
    col2.metric("Score moyen (proba défaut)", f"{metrics['score_moyen']:.4f}")
    col3.metric("Temps d'inférence moyen", f"{metrics['temps_moyen']:.1f} ms")

st.divider()

# --- Section 2 : Graphiques ---

st.subheader("Analyse des scores")

col_left, col_right = st.columns(2)

with col_left:
    if df_hist.empty:
        st.info("Données de prédictions indisponibles.")
    else:
        st.plotly_chart(build_histogram(df_hist), use_container_width=True)

with col_right:
    if df_logs_500.empty or df_preds_500.empty:
        st.info("Pas encore assez de données pour afficher l'évolution.")
    else:
        merged_500 = df_logs_500.merge(df_preds_500, on="sk_id_curr", how="inner")
        weeks = merged_500["requested_at"].apply(
            lambda x: pd.to_datetime(x).to_period("W")
        ).nunique()
        if weeks < 2:
            st.info("Pas encore assez de semaines de données pour afficher l'évolution.")
        else:
            st.plotly_chart(build_evolution_chart(merged_500), use_container_width=True)

st.divider()

# --- Section 3 : Tableau des logs ---

st.subheader("Historique des requêtes")

logs_all_resp = (
    db.table("prediction_logs")
    .select("username, sk_id_curr, requested_at, inference_time_ms, found")
    .order("requested_at", desc=True)
    .limit(50)
    .execute()
)

if not logs_all_resp.data:
    st.info("Aucun log disponible pour le moment.")
else:
    df_table = pd.DataFrame(logs_all_resp.data).rename(columns={
        "username": "Utilisateur",
        "sk_id_curr": "SK_ID_CURR",
        "requested_at": "Date / Heure",
        "inference_time_ms": "Temps (ms)",
        "found": "Client trouvé",
    })
    df_table["Temps (ms)"] = df_table["Temps (ms)"].round(2)
    df_table["Client trouvé"] = df_table["Client trouvé"].map({True: "✓", False: "✗"})
    st.dataframe(df_table, use_container_width=True, hide_index=True)
