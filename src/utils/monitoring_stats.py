import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

THRESHOLD = 0.0913


def compute_metrics(logs_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    if logs_df.empty:
        return {"taux_defaut": 0.0, "score_moyen": 0.0, "temps_moyen": 0.0, "n_clients": 0}
    merged = logs_df.merge(predictions_df, on="sk_id_curr", how="inner")
    if merged.empty:
        return {"taux_defaut": 0.0, "score_moyen": 0.0, "temps_moyen": float(logs_df["inference_time_ms"].mean()), "n_clients": 0}
    return {
        "taux_defaut": float((merged["proba_class_1"] >= THRESHOLD).mean() * 100),
        "score_moyen": float(merged["proba_class_1"].mean()),
        "temps_moyen": float(logs_df["inference_time_ms"].mean()),
        "n_clients": len(merged),
    }


def build_histogram(predictions_df: pd.DataFrame) -> go.Figure:
    df = predictions_df.copy()
    df["Risque"] = df["proba_class_1"].apply(lambda x: "Défaut" if x >= THRESHOLD else "Remboursé")
    fig = px.histogram(
        df, x="proba_class_1", color="Risque", nbins=50,
        color_discrete_map={"Remboursé": "#4e8ef7", "Défaut": "#e05c5c"},
        labels={"proba_class_1": "Score de défaut (proba_class_1)", "count": "Nombre de clients"},
        title="Distribution globale des scores",
    )
    fig.add_vline(x=THRESHOLD, line_dash="dash", line_color="#e05c5c",
                  annotation_text=f"Seuil {THRESHOLD}")
    fig.update_layout(bargap=0.05)
    return fig
