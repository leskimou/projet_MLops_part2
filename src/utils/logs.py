from src.utils.database import get_client


def log_prediction_request(user_id: int, username: str, sk_id_curr: int, inference_time_ms: float, found: bool, proba_class_1: float | None) -> None:
    client = get_client()
    client.table("prediction_logs").insert({
        "user_id": user_id,
        "username": username,
        "sk_id_curr": sk_id_curr,
        "inference_time_ms": inference_time_ms,
        "found": found,
        "proba_class_1": proba_class_1,
    }).execute()
