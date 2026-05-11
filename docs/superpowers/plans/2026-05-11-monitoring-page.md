# Monitoring Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enrichir la page monitoring Streamlit avec 3 métriques clés, un histogramme de distribution des scores et une courbe d'évolution hebdomadaire.

**Architecture:** Les fonctions de calcul et de construction des graphiques sont extraites dans `src/utils/monitoring_stats.py` (pur Python + pandas + plotly, testable unitairement). `monitoring.py` orchestre uniquement le chargement Supabase et l'affichage Streamlit.

**Tech Stack:** Streamlit, Plotly Express, pandas, Supabase Python client

---

## File Map

| Fichier | Action | Rôle |
|---|---|---|
| `pyproject.toml` | Modifier | Ajouter dépendance `plotly` |
| `src/utils/monitoring_stats.py` | Créer | Fonctions pures : métriques, figures plotly |
| `tests/unit/test_monitoring_stats.py` | Créer | Tests unitaires des fonctions pures |
| `src/api/monitoring.py` | Réécrire | Page Streamlit : chargement + affichage |

---

## Task 1 : Ajouter la dépendance plotly

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1 : Ajouter plotly dans pyproject.toml**

Dans la liste `dependencies`, ajouter après `"pandas>=3.0.2",` :

```toml
"plotly>=5.0",
```

- [ ] **Step 2 : Installer la dépendance**

```bash
uv sync
```

Expected : plotly apparaît dans `.venv`.

- [ ] **Step 3 : Vérifier l'import**

```bash
python -c "import plotly; print(plotly.__version__)"
```

Expected : un numéro de version ≥ 5.0.

- [ ] **Step 4 : Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: add plotly dependency"
```

---

## Task 2 : Créer monitoring_stats.py avec les fonctions pures [TDD]

**Files:**
- Create: `src/utils/monitoring_stats.py`
- Create: `tests/unit/test_monitoring_stats.py`

### RED — écrire les tests d'abord

- [ ] **Step 1 : Créer le fichier de tests**

Créer `tests/unit/test_monitoring_stats.py` :

```python
import pandas as pd
import pytest
from src.utils.monitoring_stats import compute_metrics, build_histogram, build_evolution_chart

THRESHOLD = 0.0913


def make_logs(n=5, inference_ms=50.0):
    return pd.DataFrame({
        "sk_id_curr": list(range(1, n + 1)),
        "inference_time_ms": [inference_ms] * n,
    })


def make_predictions(sk_ids, proba_values):
    return pd.DataFrame({
        "sk_id_curr": sk_ids,
        "proba_class_1": proba_values,
    })


# --- compute_metrics ---

def test_compute_metrics_taux_defaut_zero():
    logs = make_logs()
    preds = make_predictions([1, 2, 3, 4, 5], [0.01, 0.02, 0.03, 0.04, 0.05])
    result = compute_metrics(logs, preds)
    assert result["taux_defaut"] == 0.0


def test_compute_metrics_taux_defaut_cent_pour_cent():
    logs = make_logs()
    preds = make_predictions([1, 2, 3, 4, 5], [0.5, 0.6, 0.7, 0.8, 0.9])
    result = compute_metrics(logs, preds)
    assert result["taux_defaut"] == 100.0


def test_compute_metrics_score_moyen():
    logs = make_logs()
    preds = make_predictions([1, 2, 3, 4, 5], [0.1, 0.2, 0.3, 0.4, 0.5])
    result = compute_metrics(logs, preds)
    assert abs(result["score_moyen"] - 0.3) < 1e-9


def test_compute_metrics_temps_moyen():
    logs = make_logs(inference_ms=75.0)
    preds = make_predictions([1, 2, 3, 4, 5], [0.1] * 5)
    result = compute_metrics(logs, preds)
    assert result["temps_moyen"] == 75.0


def test_compute_metrics_retourne_zero_si_pas_de_jointure():
    logs = make_logs()
    preds = make_predictions([99, 100], [0.5, 0.6])  # sk_id_curr différents
    result = compute_metrics(logs, preds)
    assert result["n_clients"] == 0


# --- build_histogram ---

def test_build_histogram_retourne_une_figure():
    import plotly.graph_objects as go
    preds = make_predictions([1, 2, 3], [0.05, 0.15, 0.5])
    fig = build_histogram(preds)
    assert isinstance(fig, go.Figure)


def test_build_histogram_contient_deux_traces():
    preds = make_predictions([1, 2, 3], [0.05, 0.15, 0.5])
    fig = build_histogram(preds)
    # 2 traces : une par classe (Remboursé / Défaut)
    assert len(fig.data) >= 1


# --- build_evolution_chart ---

def test_build_evolution_chart_retourne_une_figure():
    import plotly.graph_objects as go
    df = pd.DataFrame({
        "requested_at": pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-19"]),
        "proba_class_1": [0.1, 0.2, 0.15],
    })
    fig = build_evolution_chart(df)
    assert isinstance(fig, go.Figure)


def test_build_evolution_chart_groupe_par_semaine():
    df = pd.DataFrame({
        "requested_at": pd.to_datetime([
            "2026-01-05", "2026-01-06",  # semaine 1
            "2026-01-12",                 # semaine 2
        ]),
        "proba_class_1": [0.1, 0.3, 0.2],
    })
    fig = build_evolution_chart(df)
    # 2 semaines = 2 points sur la courbe
    assert len(fig.data[0].x) == 2
```

- [ ] **Step 2 : Vérifier que les tests échouent (RED)**

```bash
python -m pytest tests/unit/test_monitoring_stats.py -v
```

Expected : `ModuleNotFoundError: No module named 'src.utils.monitoring_stats'`

### GREEN — implémenter les fonctions

- [ ] **Step 3 : Créer src/utils/monitoring_stats.py**

```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

THRESHOLD = 0.0913


def compute_metrics(logs_df: pd.DataFrame, predictions_df: pd.DataFrame) -> dict:
    merged = logs_df.merge(predictions_df, on="sk_id_curr", how="inner")
    if merged.empty:
        return {"taux_defaut": 0.0, "score_moyen": 0.0, "temps_moyen": logs_df["inference_time_ms"].mean(), "n_clients": 0}
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


def build_evolution_chart(merged_df: pd.DataFrame) -> go.Figure:
    df = merged_df.copy()
    df["week"] = pd.to_datetime(df["requested_at"]).dt.to_period("W").dt.start_time
    weekly = df.groupby("week")["proba_class_1"].mean().reset_index()
    weekly.columns = ["Semaine", "Score moyen"]
    fig = px.line(
        weekly, x="Semaine", y="Score moyen",
        title="Évolution du score moyen par semaine",
        markers=True,
    )
    fig.add_hline(y=THRESHOLD, line_dash="dash", line_color="#e05c5c",
                  annotation_text=f"Seuil {THRESHOLD}")
    fig.update_yaxes(range=[0, 1])
    return fig
```

- [ ] **Step 4 : Vérifier que les tests passent (GREEN)**

```bash
python -m pytest tests/unit/test_monitoring_stats.py -v
```

Expected : tous les tests `PASSED`.

- [ ] **Step 5 : Commit**

```bash
git add src/utils/monitoring_stats.py tests/unit/test_monitoring_stats.py
git commit -m "feat: add monitoring_stats helpers with tests"
```

---

## Task 3 : Réécrire monitoring.py

**Files:**
- Modify: `src/api/monitoring.py`

- [ ] **Step 1 : Réécrire monitoring.py**

Remplacer l'intégralité du fichier par :

```python
import pandas as pd
import streamlit as st

from src.utils.database import get_client
from src.utils.monitoring_stats import (
    THRESHOLD,
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
```

- [ ] **Step 2 : Lancer tous les tests unitaires**

```bash
python -m pytest tests/unit/ -v
```

Expected : tous les tests passent.

- [ ] **Step 3 : Commit**

```bash
git add src/api/monitoring.py
git commit -m "feat: rewrite monitoring page with metrics, histogram and weekly evolution chart"
```

---

## Task 4 : Vérification finale de la couverture

**Files:** aucun

- [ ] **Step 1 : Lancer la suite complète avec couverture**

```bash
python -m pytest tests/ --cov=src --cov-report=term-missing -q
```

Expected : ≥ 80% de couverture, 0 test en échec.
