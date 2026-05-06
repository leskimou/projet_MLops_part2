from unittest.mock import patch, MagicMock
from src.utils.database import get_client


def test_get_client_returns_supabase_client():
    mock_client = MagicMock()
    with patch("src.utils.database.create_client", return_value=mock_client):
        result = get_client()
    assert result is mock_client


def test_get_client_calls_create_client_once():
    mock_client = MagicMock()
    with patch("src.utils.database.create_client", return_value=mock_client) as mock_create:
        get_client()
    mock_create.assert_called_once()


def test_get_client_passes_url_and_key():
    mock_client = MagicMock()
    with patch("src.utils.database.create_client", return_value=mock_client) as mock_create:
        with patch("src.utils.database.SUPABASE_URL", "https://test.supabase.co"):
            with patch("src.utils.database.SUPABASE_KEY", "test-anon-key"):
                get_client()
    mock_create.assert_called_once_with("https://test.supabase.co", "test-anon-key")
