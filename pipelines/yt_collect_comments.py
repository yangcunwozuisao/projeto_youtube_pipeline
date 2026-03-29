# file: yt_collect_comments.py
import os
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


def load_api_key():
    """
    Tenta carregar a API key na seguinte ordem:
    1) Variável de ambiente YT_API_KEY
    2) Linha YT_API_KEY=... em .env (mesma pasta)
    3) Arquivo yt_api_key.txt contendo somente a chave
    """
    # 1) Variável de ambiente
    key = os.getenv("YT_API_KEY")
    if key:
        return key.strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 2) Arquivo .env
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("YT_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

    # 3) Arquivo yt_api_key.txt
    txt_path = os.path.join(base_dir, "yt_api_key.txt")
    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            return f.read().strip()

    raise RuntimeError(
        "Não encontrei a chave da API do YouTube.\n"
        "- Defina a variável de ambiente YT_API_KEY, ou\n"
        "- Crie um .env com YT_API_KEY=..., ou\n"
        "- Crie um arquivo yt_api_key.txt com a chave (uma linha)."
    )


API_KEY = load_api_key()
MAX_PER_VIDEO = 200


def client():
    """Cria cliente da API do YouTube."""
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def fetch_comments(yt, vid, limit=200):
    """
    Busca até `limit` comentários (threads) de um vídeo.
    Retorna a lista de commentThreads brutos da API.
    """
    out = []
    token = None
    got = 0

    while got < limit:
        resp = (
            yt.commentThreads()
            .list(
                part="snippet,replies",
                videoId=vid,
                maxResults=min(100, limit - got),
                textFormat="plainText",
                order="relevance",
                pageToken=token,
            )
            .execute()
        )

        batch = resp.get("items", [])
        out.extend(batch)
        got += len(batch)

        token = resp.get("nextPageToken")
        if not token or not batch:
            break

        time.sleep(0.1)

    return out


def flatten(video_id, thread):
    """
    Achata um commentThread em linhas (top-level + replies).
    Retorna lista de dicionários.
    """
    rows = []

    top = thread.get("snippet", {}).get("topLevelComment", {})
    ts = top.get("snippet", {})

    if ts:
        rows.append(
            {
                "videoId": video_id,
                "commentId": top.get("id"),
                "author": ts.get("authorDisplayName"),
                "text": ts.get("textDisplay"),  # se quiser texto “puro”, use textOriginal
                "likeCount": ts.get("likeCount"),
                "publishedAt": ts.get("publishedAt"),
                "isReply": False,
                "replyTo": None,
            }
        )

    replies_container = (thread.get("replies") or {})
    for r in replies_container.get("comments", []) or []:
        rs = r.get("snippet", {})
        rows.append(
            {
                "videoId": video_id,
                "commentId": r.get("id"),
                "author": rs.get("authorDisplayName"),
                "text": rs.get("textDisplay"),
                "likeCount": rs.get("likeCount"),
                "publishedAt": rs.get("publishedAt"),
                "isReply": True,
                "replyTo": top.get("id") if top else None,
            }
        )

    return rows


def main():
    yt = client()

    # Lê lista de vídeos do arquivo videos.csv (coluna: videoId)
    vids = (
        pd.read_csv("outputs/videos.csv")["videoId"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    all_rows = []

    for vid in tqdm(vids, desc="Coletando comentários"):
        try:
            threads = fetch_comments(yt, vid, MAX_PER_VIDEO)
            for t in threads:
                all_rows.extend(flatten(vid, t))
        except HttpError as e:
            print(f"[warn] {vid} sem comentários ou restrito: {e}")

    if all_rows:
        df = pd.DataFrame(all_rows)
        df.to_csv("outputs/comments.csv", index=False, encoding="utf-8-sig")
        print(f" Salvo comments.csv com {len(all_rows)} linhas")
    else:
        print("⚠ Nenhum comentário coletado.")


if __name__ == "__main__":
    main()
