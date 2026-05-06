import streamlit as st
from src.utils.database import get_client

st.title("Prédiction de crédit client")

sk_id = st.text_input("Numéro du client (SK_ID_CURR)")

if st.button("Rechercher"):
    if not sk_id or not sk_id.strip().isdigit():
        st.warning("Veuillez entrer un numéro de client valide.")
        st.stop()

    client = get_client()
    response = (
        client.table("predictions")
        .select("predicted_class, proba_class_0, proba_class_1")
        .eq("sk_id_curr", int(sk_id))
        .execute()
    )

    if not response.data:
        st.warning(f"Aucune prédiction trouvée pour le client {int(sk_id)}.")
    else:
        row = response.data[0]
        predicted = int(row["predicted_class"])
        proba_0 = float(row["proba_class_0"])
        proba_1 = float(row["proba_class_1"])

        st.subheader(f"Résultats pour le client {int(sk_id)}")

        if predicted == 0:
            st.success(f"Classe prédite : {predicted} — Crédit remboursé")
        else:
            st.error(f"Classe prédite : {predicted} — Defaut de remboursement")

        st.metric("Probabilité classe 0 (remboursé)", f"{proba_0:.2%}")
        st.metric("Probabilité classe 1 (défaut de remboursement)", f"{proba_1:.2%}")
