import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv("config/dev/.env")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


def get_client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


if __name__ == "__main__":
    client = get_client()
    print("Connexion Supabase OK :", client)
