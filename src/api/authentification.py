import streamlit as st
from src.utils.auth import authenticate

st.title("Prédiction de crédit client")
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
