import time

import streamlit as st

from src.utils.database import get_client

st.title("Monitoring")

user = st.session_state.user

if user is None or user.get("role") != "administrateur":
    st.error("Accès refusé. Cette page est réservée aux administrateurs.")
    st.stop()

st.subheader("Temps d'inférence")
st.write("Mesurez le temps de récupération d'une prédiction depuis la base de données.")

sk_id = st.text_input("Numéro du client (SK_ID_CURR)")

if st.button("Mesurer"):
    if not sk_id or not sk_id.strip().isdigit():
        st.warning("Veuillez entrer un numéro de client valide.")
        st.stop()

    db_client = get_client()

    start = time.perf_counter()
    response = (
        db_client.table("predictions")
        .select("predicted_class, proba_class_0, proba_class_1")
        .eq("sk_id_curr", int(sk_id))
        .execute()
    )
    elapsed_ms = (time.perf_counter() - start) * 1000

    st.metric("Temps d'inférence", f"{elapsed_ms:.2f} ms")

    if response.data:
        st.success(f"Prédiction récupérée pour le client {int(sk_id)}.")
    else:
        st.warning(f"Aucune prédiction trouvée pour le client {int(sk_id)}.")
