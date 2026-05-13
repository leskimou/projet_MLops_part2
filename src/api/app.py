import streamlit as st

if "user" not in st.session_state:
    st.session_state.user = None

user = st.session_state.user

if user is None:
    pg = st.navigation([st.Page("authentification.py", title="Connexion")])
else:
    role_labels = {
        "administrateur": "Administrateur",

    }
    role_label = role_labels.get(user["role"], "Client")

    st.sidebar.success(f"Connecté : **{user['username']}**")
    st.sidebar.caption(f"Rôle : {role_label}")

    if st.sidebar.button("Se déconnecter"):
        st.session_state.user = None
        st.rerun()

    pages = [st.Page("prediction.py", title="Prédiction")]
    if user["role"] == "administrateur":
        pages.append(st.Page("monitoring.py", title="Monitoring"))

    pg = st.navigation(pages)

pg.run()
