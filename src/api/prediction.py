import time

import streamlit as st

from src.utils.database import get_client
from src.utils.logs import log_prediction_request

st.title("Prédiction remboursement de crédit")

user = st.session_state.user

sk_id = st.text_input("Numéro du client (SK_ID_CURR)")

if st.button("Rechercher"):
    if not sk_id or not sk_id.strip().isdigit() or int(sk_id) > 2_147_483_647:
        st.error("Numéro de client invalide. Saisissez un identifiant numérique entier (ex : 100002).")
        st.stop()

    client = get_client()
    start = time.perf_counter()
    response = (
        client.table("predictions")
        .select("proba_class_0, proba_class_1")
        .eq("sk_id_curr", int(sk_id))
        .execute()
    )
    inference_time_ms = (time.perf_counter() - start) * 1000
    found = bool(response.data)
    proba_class_1 = float(response.data[0]["proba_class_1"]) if found else None

    log_prediction_request(
        user_id=user["id"],
        username=user["username"],
        sk_id_curr=int(sk_id),
        inference_time_ms=inference_time_ms,
        found=found,
        proba_class_1=proba_class_1,
    )

    if not found:
        st.error(f"Client n° {int(sk_id)} introuvable. Vérifiez l'identifiant et réessayez.")
    else:
        row = response.data[0]
        proba_0 = float(row["proba_class_0"])
        proba_1 = float(row["proba_class_1"])
        predicted = 1 if proba_1 >= 0.0913 else 0

        st.subheader(f"Résultats pour le client {int(sk_id)}")

        if predicted == 0:
            st.success(f"Classe prédite : {predicted} — Crédit remboursé")
        else:
            st.error(f"Classe prédite : {predicted} — Défaut de remboursement")

        st.metric("Probabilité classe 0 (remboursé)", f"{proba_0:.2%}")
        st.metric("Probabilité classe 1 (défaut de remboursement)", f"{proba_1:.2%}")

        if user["role"] == "administrateur":
            st.divider()
            st.caption(f"Debug — SK_ID: {sk_id} | Rôle: {user['role']}")
