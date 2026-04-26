"""
db.py — Conexão singleton com o MongoDB.

MONGO_URI pode ser configurada via variável de ambiente, evitando
hard-code de "localhost" em ambientes de produção/Docker.
"""

from __future__ import annotations

import os

from pymongo import MongoClient
from pymongo.database import Database

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB  = os.getenv("MONGO_DB",  "youtube_db")

_client: MongoClient | None = None


def get_db() -> Database:
    global _client
    if _client is None:
        print(f"[db] Conectando MongoDB ({MONGO_URI})...")
        _client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    return _client[MONGO_DB]


def close_db() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None