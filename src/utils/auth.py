import bcrypt
from src.utils.database import get_client


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def get_user(username: str) -> dict | None:
    client = get_client()
    response = (
        client.table("users")
        .select("id, username, password_hash, role")
        .eq("username", username)
        .execute()
    )
    return response.data[0] if response.data else None


def authenticate(username: str, password: str) -> dict | None:
    user = get_user(username)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None
