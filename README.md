---
title: Credit Prediction
emoji: 💳
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Prédiction de remboursement de crédit

Application MLOps de scoring crédit déployée sur Hugging Face Spaces. Elle prédit la probabilité qu'un client rembourse ou non son prêt, à partir d'un modèle CatBoost entraîné sur le dataset Home Credit.

## Objectif

L'API permet à des conseillers bancaires de consulter le score de risque d'un client en saisissant son identifiant (`SK_ID_CURR`). Les prédictions sont calculées en batch en amont et stockées en base de données, l'application se limite à interroger la DB et à afficher le résultat avec le seuil de décision optimisé lors de la phase de modèlisation (0.0913).

Un espace d'administration donne accès au monitoring de l'utilisation de l'API (logs de requêtes, statistiques).

## Architecture

```
src/
├── api/
│   ├── app.py               # Point d'entrée Streamlit (navigation, état session)
│   ├── authentification.py  # Page de connexion
│   ├── prediction.py        # Page de recherche et affichage du score
│   └── monitoring.py        # Tableau de bord admin (logs, stats)
├── models/
│   ├── predict.py           # Pipeline batch : inférence CatBoost → Supabase
│   └── export_onnx.py       # Export du modèle au format ONNX
├── data/
│   └── preprocessing.py     # Prétraitement des données brutes
└── utils/
    ├── database.py          # Client Supabase
    ├── auth.py              # Gestion de l'authentification
    ├── logs.py              # Journalisation des requêtes de prédiction
    └── monitoring_stats.py  # Agrégation des statistiques de monitoring
docs/
├── getstarted.ipynb         # Guide d'utilisation de l'API
├── datadrift.ipynb          # Analyse du data drift
└── optimization_report.ipynb # Benchmark CatBoost vs ONNX Runtime
```

**Base de données (Supabase) — 3 tables :**
- `users` : comptes autorisés à accéder à l'API (rôles : client / administrateur)
- `predictions` : scores des deux classes pour chaque `SK_ID_CURR`, calculés en batch
- `predictions_logs` : historique des requêtes (utilisateur, identifiant consulté, temps d'inférence)

## Notebooks

### `docs/getstarted.ipynb` — Guide d'utilisation

Présentation pas à pas de l'application :
- Authentification via identifiant/mot de passe
- Recherche d'un client par son `SK_ID_CURR` (plage de test : 100002 → 456255)
- Interprétation des résultats (classe prédite, probabilités, seuil de décision)
- Accès à l'interface de monitoring pour les comptes administrateur
- Schéma et description des trois tables Supabase

### `docs/datadrift.ipynb` — Analyse du data drift

Étude du glissement de distribution des données entre les données de référence et les données de production simulées :
- Les 70 % de données les plus anciennes (par `SK_ID_CURR`) servent de référence, les 30 % restants simulent des données de production
- Rapport de drift généré avec **Evidently** (`DataDriftPreset`) sur l'ensemble des features
- Visualisation de la distribution de `SK_ID_CURR` pour valider le découpage temporel

### `docs/optimization_report.ipynb` — Rapport d'optimisation

Benchmark du pipeline d'inférence batch CatBoost → ONNX Runtime :
- Profiling avec `cProfile` pour identifier les goulots d'étranglement (`predict_proba` / conversion interne Python-CatBoost)
- Export du modèle au format ONNX via `export_onnx.py`
- Validation numérique : delta max entre CatBoost et ONNX < 1e-5 (aucune régression de précision)
- Benchmark sur 1 000 lignes × 100 itérations — ONNX Runtime : **1.14× plus rapide** (11.55 ms vs 13.13 ms)
