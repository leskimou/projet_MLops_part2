# Monitoring Page — Design Spec
**Date:** 2026-05-11

## Objectif

Enrichir la page `monitoring.py` avec trois nouvelles sections d'analyse des scores de prédiction, en conservant le tableau des logs existant.

---

## Layout

Page unique scrollable (pas d'onglets). Quatre sections de haut en bas :

1. Métriques clés
2. Analyse des scores (deux graphiques côte à côte)
3. Historique des requêtes (tableau existant)

---

## Section 1 — Métriques clés

Trois `st.metric` en colonnes, calculés sur les **50 derniers logs où `found=True`** via un JOIN `prediction_logs` × `predictions` sur `sk_id_curr`.

| Métrique | Calcul |
|---|---|
| Taux de défaut prédit | % de clients consultés avec `proba_class_1 >= 0.0913` |
| Score moyen | Moyenne de `proba_class_1` des clients consultés |
| Temps d'inférence moyen | Moyenne de `inference_time_ms` (déjà en base, pas de JOIN) |

Si aucun log `found=True` disponible : afficher `st.info("Pas encore assez de données.")` et arrêter.

---

## Section 2 — Analyse des scores

### Graphique gauche : Distribution globale des scores

- **Source :** table `predictions`, colonne `proba_class_1`, tous les clients (307 506 lignes)
- **Type :** histogramme Plotly (`px.histogram`)
- **Rendu :** `st.plotly_chart(..., use_container_width=True)`
- **Détails :** ligne verticale rouge pointillée au seuil `0.0913`, légende "Remboursé / Défaut" en couleurs distinctes

### Graphique droit : Évolution du score moyen par semaine

- **Source :** JOIN `prediction_logs` (WHERE `found=True`) × `predictions` sur `sk_id_curr`
- **Regroupement :** par semaine ISO (`requested_at` tronqué à la semaine)
- **Type :** courbe Plotly (`px.line`)
- **Rendu :** `st.plotly_chart(..., use_container_width=True)`
- **Détails :** ligne horizontale rouge pointillée au seuil `0.0913`

Si moins de 2 semaines de données : afficher `st.info("Pas encore assez de données pour afficher l'évolution.")`.

---

## Section 3 — Historique des requêtes

Tableau existant (inchangé) : 50 derniers logs, colonnes `Utilisateur`, `SK_ID_CURR`, `Date / Heure`, `Temps (ms)`, `Client trouvé` (✓ / ✗).

---

## Dépendances

- **Plotly** : à ajouter dans `pyproject.toml` (`plotly>=5.0`)
- **JOIN Supabase** : pas de JOIN natif côté client — charger les deux tables séparément en Python et merger avec pandas

---

## Données à charger

```
predictions     → colonne proba_class_1 uniquement, tous les clients (histogramme)
prediction_logs → 50 derniers WHERE found=True → métriques (taux défaut, score moy.)
prediction_logs → 500 derniers WHERE found=True + JOIN predictions → évolution hebdo
prediction_logs → 50 derniers tous (tableau logs)
```

La limite de 500 pour l'évolution garantit plusieurs semaines de données sans surcharger Supabase.

---

## Fichiers modifiés

| Fichier | Changement |
|---|---|
| `src/api/monitoring.py` | Réécriture complète |
| `pyproject.toml` | Ajout dépendance `plotly` |
