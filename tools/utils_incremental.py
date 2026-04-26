from db import get_db

def get_last_published_at():
    db = get_db()

    doc = db["videos"].find_one(
        {},
        sort=[("publishedAt", -1)]
    )

    if not doc:
        print("Nenhum dado anterior encontrado -> coleta completa")
        return None

    last_date = doc.get("publishedAt")

    if not last_date:
        print("Nenhum campo publishedAt encontrado no banco -> coleta completa")
        return None

    print(f"[INFO] Data do vídeo mais recente no banco: {last_date}")
    return last_date