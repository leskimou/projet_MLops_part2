from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = "src/api/app.py"


def make_mock_client(data):
    mock_response = MagicMock()
    mock_response.data = data

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    return mock_client


# --- Tests de rendu initial ---

def test_app_affiche_le_titre():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert at.title[0].value == "Prédiction de crédit client"


def test_app_affiche_champ_texte():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert len(at.text_input) == 1
    assert "SK_ID_CURR" in at.text_input[0].label


def test_app_affiche_bouton_rechercher():
    at = AppTest.from_file(APP_PATH)
    at.run()
    assert at.button[0].label == "Rechercher"


# --- Tests de validation ---

def test_app_avertit_si_champ_vide():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.button[0].click()
    at.run()
    assert any("valide" in w.value for w in at.warning)


def test_app_avertit_si_saisie_non_numerique():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("abc")
    at.button[0].click()
    at.run()
    assert any("valide" in w.value for w in at.warning)


def test_app_avertit_si_saisie_decimale():
    at = AppTest.from_file(APP_PATH)
    at.run()
    at.text_input[0].set_value("123.45")
    at.button[0].click()
    at.run()
    assert any("valide" in w.value for w in at.warning)


# --- Tests fonctionnels avec mock Supabase ---

def test_app_client_trouve_credit_rembourse():
    mock_client = make_mock_client([{
        "predicted_class": 0,
        "proba_class_0": 0.85,
        "proba_class_1": 0.15,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].set_value("100001")
        at.button[0].click()
        at.run()

    assert any("remboursé" in s.value for s in at.success)
    assert len(at.metric) == 2


def test_app_client_trouve_defaut_remboursement():
    mock_client = make_mock_client([{
        "predicted_class": 1,
        "proba_class_0": 0.2,
        "proba_class_1": 0.8,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].set_value("100002")
        at.button[0].click()
        at.run()

    assert any("Defaut" in e.value for e in at.error)
    assert len(at.metric) == 2


def test_app_client_non_trouve():
    mock_client = make_mock_client([])
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].set_value("999999")
        at.button[0].click()
        at.run()

    assert any("Aucune prédiction" in w.value for w in at.warning)


def test_app_affiche_les_deux_metriques():
    mock_client = make_mock_client([{
        "predicted_class": 0,
        "proba_class_0": 0.72,
        "proba_class_1": 0.28,
    }])
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file(APP_PATH)
        at.run()
        at.text_input[0].set_value("100003")
        at.button[0].click()
        at.run()

    labels = [m.label for m in at.metric]
    assert any("remboursé" in label for label in labels)
    assert any("défaut" in label for label in labels)
