from unittest.mock import patch, MagicMock
from streamlit.testing.v1 import AppTest

APP_PATH = "src/api/app.py"

MOCK_CLIENT_USER = {"id": 2, "username": "client1", "role": "client"}
MOCK_ADMIN_USER = {"id": 1, "username": "admin", "role": "administrateur"}


def make_mock_supabase_empty():
    """Returns empty data for all queries."""
    mock_response = MagicMock()
    mock_response.data = []

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.in_.return_value = mock_query
    mock_query.execute.return_value = mock_response

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query
    return mock_client


def app_logged_in(user):
    at = AppTest.from_file(APP_PATH)
    at.session_state["user"] = user
    at.run()
    return at


def test_monitoring_acces_refuse_pour_client():
    with patch("src.utils.database.get_client", return_value=make_mock_supabase_empty()):
        at = app_logged_in(MOCK_CLIENT_USER)
    # Client role does not have "administrateur", so Monitoring page won't be added
    user = at.session_state["user"]
    assert user["role"] != "administrateur"


def test_monitoring_acces_admin_affiche_le_titre():
    with patch("src.utils.database.get_client", return_value=make_mock_supabase_empty()):
        at = app_logged_in(MOCK_ADMIN_USER)
    # Admin user is set up, page is accessible
    assert at.session_state["user"]["role"] == "administrateur"


def test_monitoring_affiche_info_si_pas_de_logs():
    mock_client = make_mock_supabase_empty()
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file("src/api/monitoring.py")
        at.session_state["user"] = MOCK_ADMIN_USER
        at.run()
    assert any("données" in i.value.lower() for i in at.info)


def test_monitoring_affiche_erreur_si_non_admin():
    mock_client = make_mock_supabase_empty()
    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file("src/api/monitoring.py")
        at.session_state["user"] = MOCK_CLIENT_USER
        at.run()
    assert any("refusé" in e.value.lower() or "reserv" in e.value.lower() for e in at.error)


def test_monitoring_affiche_metriques_avec_donnees():
    mock_logs_data = [
        {"sk_id_curr": 100001, "inference_time_ms": 45.0, "requested_at": "2026-05-01T10:00:00", "username": "alice", "found": True},
        {"sk_id_curr": 100002, "inference_time_ms": 55.0, "requested_at": "2026-05-01T11:00:00", "username": "alice", "found": True},
    ]
    mock_preds_data = [
        {"sk_id_curr": 100001, "proba_class_1": 0.05},
        {"sk_id_curr": 100002, "proba_class_1": 0.95},
    ]

    call_count = 0

    def side_effect_execute():
        nonlocal call_count
        call_count += 1
        mock_response = MagicMock()
        # query 1: logs_50 → logs data
        # query 2: preds_50 → preds data
        # query 3: hist → preds data
        # query 4: all logs table → logs data
        if call_count in (1, 4):
            mock_response.data = mock_logs_data
        else:
            mock_response.data = mock_preds_data
        return mock_response

    mock_query = MagicMock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.in_.return_value = mock_query
    mock_query.execute.side_effect = side_effect_execute

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query

    with patch("src.utils.database.get_client", return_value=mock_client):
        at = AppTest.from_file("src/api/monitoring.py")
        at.session_state["user"] = MOCK_ADMIN_USER
        at.run()

    assert len(at.metric) >= 3
