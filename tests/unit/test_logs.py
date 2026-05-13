from unittest.mock import patch, MagicMock
from src.utils.logs import log_prediction_request


def make_mock_supabase():
    mock_query = MagicMock()
    mock_query.insert.return_value = mock_query
    mock_query.execute.return_value = MagicMock()

    mock_client = MagicMock()
    mock_client.table.return_value = mock_query
    return mock_client


def test_log_prediction_request_insere_dans_prediction_logs():
    mock_client = make_mock_supabase()
    with patch("src.utils.logs.get_client", return_value=mock_client):
        log_prediction_request(user_id=1, username="alice", sk_id_curr=100001, inference_time_ms=42.5, found=True, proba_class_1=0.05)
    mock_client.table.assert_called_once_with("prediction_logs")


def test_log_prediction_request_insere_found_true_avec_proba():
    mock_client = make_mock_supabase()
    with patch("src.utils.logs.get_client", return_value=mock_client):
        log_prediction_request(user_id=7, username="bob", sk_id_curr=999, inference_time_ms=123.4, found=True, proba_class_1=0.42)
    mock_client.table.return_value.insert.assert_called_once_with({
        "user_id": 7,
        "username": "bob",
        "sk_id_curr": 999,
        "inference_time_ms": 123.4,
        "found": True,
        "proba_class_1": 0.42,
    })


def test_log_prediction_request_insere_found_false_avec_proba_none():
    mock_client = make_mock_supabase()
    with patch("src.utils.logs.get_client", return_value=mock_client):
        log_prediction_request(user_id=3, username="carol", sk_id_curr=0, inference_time_ms=5.1, found=False, proba_class_1=None)
    mock_client.table.return_value.insert.assert_called_once_with({
        "user_id": 3,
        "username": "carol",
        "sk_id_curr": 0,
        "inference_time_ms": 5.1,
        "found": False,
        "proba_class_1": None,
    })


def test_log_prediction_request_execute_la_requete():
    mock_client = make_mock_supabase()
    with patch("src.utils.logs.get_client", return_value=mock_client):
        log_prediction_request(user_id=1, username="alice", sk_id_curr=100001, inference_time_ms=10.0, found=True, proba_class_1=0.08)
    mock_client.table.return_value.insert.return_value.execute.assert_called_once()
