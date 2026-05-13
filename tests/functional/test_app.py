from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = "src/api/app.py"

MOCK_CLIENT_USER = {"id": 2, "username": "client1", "role": "client"}
MOCK_ADMIN_USER = {"id": 1, "username": "admin", "role": "administrateur"}

THRESHOLD = 0.0913


def make_mock_supabase(data):
    mock_response = MagicMock()
    mock_response.data = data

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query
    return mock_client


def app_logged_in(user):
    at = AppTest.from_file(APP_PATH)
    at.session_state["user"] = user
    at.run()
    return at


# --- Page de login (utilisateur non connecté) ---

def test_login_affiche_le_titre():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert at.title[0].value == "Prédiction remboursement de crédit"


def test_login_affiche_sous_titre_connexion():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert any("Connexion" in s.value for s in at.subheader)


def test_login_affiche_deux_champs_texte():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.text_input) == 2


def test_login_affiche_bouton_se_connecter():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert any(b.label == "Se connecter" for b in at.button)


def test_login_avertit_si_champs_vides():
    at = AppTest.from_file(APP_PATH)
    at.run()
    next(b for b in at.button if b.label == "Se connecter").click()
    at.run()
    assert any("remplir" in w.value for w in at.warning)


def test_login_erreur_si_identifiants_incorrects():
    with patch("src.utils.auth.authenticate", return_value=None):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].set_value("mauvaisuser")
        at.text_input[1].set_value("mauvaismdp")
        next(b for b in at.button if b.label == "Se connecter").click()
        at.run()
    assert any("incorrects" in e.value for e in at.error)


# --- Application principale (utilisateur connecté) ---

def test_app_affiche_le_titre():
    at = app_logged_in(MOCK_CLIENT_USER)
    assert at.title[0].value == "Prédiction remboursement de crédit"


def test_app_affiche_champ_sk_id():
    at = app_logged_in(MOCK_CLIENT_USER)
    assert len(at.text_input) == 1
    assert "SK_ID_CURR" in at.text_input[0].label


def test_app_affiche_bouton_rechercher():
    at = app_logged_in(MOCK_CLIENT_USER)
    assert any(b.label == "Rechercher" for b in at.button)


def test_app_erreur_si_champ_vide():
    at = app_logged_in(MOCK_CLIENT_USER)
    next(b for b in at.button if b.label == "Rechercher").click()
    at.run()
    assert any("invalide" in e.value for e in at.error)


def test_app_erreur_si_saisie_non_numerique():
    at = app_logged_in(MOCK_CLIENT_USER)
    at.text_input[0].set_value("abc")
    next(b for b in at.button if b.label == "Rechercher").click()
    at.run()
    assert any("invalide" in e.value for e in at.error)


def test_app_erreur_si_saisie_decimale():
    at = app_logged_in(MOCK_CLIENT_USER)
    at.text_input[0].set_value("123.45")
    next(b for b in at.button if b.label == "Rechercher").click()
    at.run()
    assert any("invalide" in e.value for e in at.error)


def test_app_erreur_si_entier_hors_limite():
    at = app_logged_in(MOCK_CLIENT_USER)
    at.text_input[0].set_value("56586613315")
    next(b for b in at.button if b.label == "Rechercher").click()
    at.run()
    assert any("invalide" in e.value for e in at.error)


# proba_class_1 < THRESHOLD → classe 0 (remboursé)
def test_app_client_trouve_credit_rembourse():
    mock_client = make_mock_supabase([{
        "proba_class_0": 0.97,
        "proba_class_1": 0.03,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100001")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("remboursé" in s.value for s in at.success)
    assert len(at.metric) == 2


# proba_class_1 >= THRESHOLD → classe 1 (défaut)
def test_app_client_trouve_defaut_remboursement():
    mock_client = make_mock_supabase([{
        "proba_class_0": 0.2,
        "proba_class_1": 0.8,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100002")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("Défaut" in e.value for e in at.error)
    assert len(at.metric) == 2


def test_app_client_non_trouve():
    mock_client = make_mock_supabase([])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("999999")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("introuvable" in e.value for e in at.error)


def test_app_affiche_les_deux_metriques():
    mock_client = make_mock_supabase([{
        "proba_class_0": 0.72,
        "proba_class_1": 0.28,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100003")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    labels = [m.label for m in at.metric]
    assert any("remboursé" in label for label in labels)
    assert any("défaut" in label for label in labels)


# Seuil exact : proba juste en dessous → remboursé
def test_seuil_juste_en_dessous_donne_credit_rembourse():
    mock_client = make_mock_supabase([{
        "proba_class_0": 1 - 0.09,
        "proba_class_1": 0.09,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100004")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("remboursé" in s.value for s in at.success)


# Seuil exact : proba égale au seuil → défaut
def test_seuil_egal_donne_defaut():
    mock_client = make_mock_supabase([{
        "proba_class_0": 1 - THRESHOLD,
        "proba_class_1": THRESHOLD,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100005")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("Défaut" in e.value for e in at.error)


def test_app_administrateur_voit_section_debug():
    mock_client = make_mock_supabase([{
        "proba_class_0": 0.9,
        "proba_class_1": 0.1,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_ADMIN_USER)
        at.text_input[0].set_value("100001")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert any("Debug" in c.value for c in at.caption)


def test_app_client_ne_voit_pas_section_debug():
    mock_client = make_mock_supabase([{
        "proba_class_0": 0.9,
        "proba_class_1": 0.1,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client), \
         patch("src.utils.logs.get_client", return_value=mock_client):
        at = app_logged_in(MOCK_CLIENT_USER)
        at.text_input[0].set_value("100001")
        next(b for b in at.button if b.label == "Rechercher").click()
        at.run()
    assert not any("Debug" in c.value for c in at.caption)
