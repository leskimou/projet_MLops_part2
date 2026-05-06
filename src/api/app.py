import streamlit as st
from src.utils.database import get_client
from src.utils.auth import authenticate

st.title("Prédiction de crédit client")

# --- Gestion de la session ---
if "user" not in st.session_state:
    st.session_state.user = None

# --- Page de connexion ---
if st.session_state.user is None:
    st.subheader("Connexion")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")

    if st.button("Se connecter"):
        if not username or not password:
            st.warning("Veuillez remplir tous les champs.")
        else:
            user = authenticate(username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            else:
                st.error("Identifiants incorrects.")
    st.stop()

# --- Utilisateur connecté ---
user = st.session_state.user
role_label = "Développeur" if user["role"] == "developpeur" else "Client"
st.sidebar.success(f"Connecté : **{user['username']}**")
st.sidebar.caption(f"Rôle : {role_label}")

if st.sidebar.button("Se déconnecter"):
    st.session_state.user = None
    st.rerun()

# --- Application principale ---
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
            st.error(f"Classe prédite : {predicted} — Défaut de remboursement")

        st.metric("Probabilité classe 0 (remboursé)", f"{proba_0:.2%}")
        st.metric("Probabilité classe 1 (défaut de remboursement)", f"{proba_1:.2%}")

        # Informations techniques réservées aux développeurs
        if user["role"] == "developpeur":
            st.divider()
            st.caption(f"Debug — SK_ID: {sk_id} | Rôle: {user['role']}")
