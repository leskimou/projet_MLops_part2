import bcrypt
from unittest.mock import patch, MagicMock
from src.utils.auth import verify_password, get_user, authenticate


def make_mock_supabase(user_data):
    mock_response = MagicMock()
    mock_response.data = user_data

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query
    return mock_client


def make_user(role="client", password="secret"):
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    return {"id": 1, "username": "testuser", "password_hash": hashed, "role": role}


# --- verify_password ---

def test_verify_password_correct():
    user = make_user(password="monmotdepasse")
    assert verify_password("monmotdepasse", user["password_hash"]) is True


def test_verify_password_incorrect():
    user = make_user(password="correct")
    assert verify_password("mauvais", user["password_hash"]) is False


def test_verify_password_sensible_a_la_casse():
    user = make_user(password="Secret")
    assert verify_password("secret", user["password_hash"]) is False


# --- get_user ---

def test_get_user_retourne_utilisateur_si_trouve():
    user = make_user()
    mock_client = make_mock_supabase([user])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = get_user("testuser")
    assert result == user


def test_get_user_retourne_none_si_non_trouve():
    mock_client = make_mock_supabase([])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = get_user("inconnu")
    assert result is None


def test_get_user_interroge_la_bonne_table():
    mock_client = make_mock_supabase([])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        get_user("testuser")
    mock_client.table.assert_called_once_with("users")


# --- authenticate ---

def test_authenticate_succes_client():
    user = make_user(role="client", password="mdpclient")
    mock_client = make_mock_supabase([user])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = authenticate("testuser", "mdpclient")
    assert result == user


def test_authenticate_succes_developpeur():
    user = make_user(role="developpeur", password="mdpdev")
    mock_client = make_mock_supabase([user])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = authenticate("testuser", "mdpdev")
    assert result["role"] == "developpeur"


def test_authenticate_echec_mauvais_mot_de_passe():
    user = make_user(password="correct")
    mock_client = make_mock_supabase([user])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = authenticate("testuser", "mauvais")
    assert result is None


def test_authenticate_echec_utilisateur_inconnu():
    mock_client = make_mock_supabase([])
    with patch("src.utils.auth.get_client", return_value=mock_client):
        result = authenticate("inconnu", "quelconque")
    assert result is None
