from pymongo import MongoClient

_client = None

def get_db():
    global _client

    if _client is None:
        print("🔌 Conectando MongoDB (uma única vez)...")
        _client = MongoClient("mongodb://localhost:27017/")

    return _client["youtube_db"]