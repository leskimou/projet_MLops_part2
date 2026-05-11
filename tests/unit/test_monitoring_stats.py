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
    preds = make_predictions([99, 100], [0.5, 0.6])  # different sk_id_curr
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
            "2026-01-05", "2026-01-06",  # week 1
            "2026-01-12",                 # week 2
        ]),
        "proba_class_1": [0.1, 0.3, 0.2],
    })
    fig = build_evolution_chart(df)
    # 2 weeks = 2 points on the curve
    assert len(fig.data[0].x) == 2
