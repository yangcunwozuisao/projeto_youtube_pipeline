"""
yt_collect_comments.py — Coleta comentários dos vídeos via YouTube Data API v3.

Correção: API_KEY era carregada no topo do módulo, causando crash imediato
ao fazer `from pipelines import yt_collect_comments` mesmo sem chamar main().
Agora é carregada apenas dentro de main() e client().
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm


def load_api_key() -> str:
    """
    Carrega a API key na seguinte ordem:
    1) Variável de ambiente YT_API_KEY
    2) Linha YT_API_KEY=... em .env (mesma pasta)
    3) Arquivo yt_api_key.txt contendo somente a chave
    """
    key = os.getenv("YT_API_KEY")
    if key:
        return key.strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("YT_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")

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


MAX_PER_VIDEO = 200

VIDEO_SOURCES = [
    ("outputs/videos_top50.csv",  "top 50 por views (consistente com ASR/NLP)"),
    ("outputs/videos_clean.csv",  "dataset limpo (step 2)"),
    ("outputs/videos.csv",        "coleta bruta (step 1)"),
]


def _build_client(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def load_video_ids() -> list[str]:
    for path, label in VIDEO_SOURCES:
        if Path(path).exists():
            print(f"[info] lendo vídeos de: {path} ({label})")
            return (
                pd.read_csv(path)["videoId"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

    raise FileNotFoundError(
        "Nenhum arquivo de vídeos encontrado. Execute collect e filter primeiro.\n"
        "Esperados (em ordem de preferência):\n"
        + "\n".join(f"  • {p}" for p, _ in VIDEO_SOURCES)
    )


def fetch_comments(yt, vid: str, limit: int = 200) -> list:
    out   = []
    token = None
    got   = 0

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


def flatten(video_id: str, thread: dict) -> list[dict]:
    rows = []

    top = thread.get("snippet", {}).get("topLevelComment", {})
    ts  = top.get("snippet", {})

    if ts:
        rows.append({
            "videoId":     video_id,
            "commentId":   top.get("id"),
            "author":      ts.get("authorDisplayName"),
            "text":        ts.get("textDisplay"),
            "likeCount":   ts.get("likeCount"),
            "publishedAt": ts.get("publishedAt"),
            "isReply":     False,
            "replyTo":     None,
        })

    replies_container = thread.get("replies") or {}
    for r in replies_container.get("comments", []) or []:
        rs = r.get("snippet", {})
        rows.append({
            "videoId":     video_id,
            "commentId":   r.get("id"),
            "author":      rs.get("authorDisplayName"),
            "text":        rs.get("textDisplay"),
            "likeCount":   rs.get("likeCount"),
            "publishedAt": rs.get("publishedAt"),
            "isReply":     True,
            "replyTo":     top.get("id") if top else None,
        })

    return rows


def load_existing_comments() -> set[str]:
    path = Path("outputs/comments.csv")
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path)
        return set(existing["videoId"].dropna().astype(str).unique().tolist())
    except Exception as e:
        print(f"[warn] não consegui ler comments.csv existente: {e}")
        return set()


def main() -> None:
    # Carrega a API key apenas ao executar, não no import
    api_key = load_api_key()
    yt      = _build_client(api_key)

    vids = load_video_ids()

    already_collected = load_existing_comments()
    pending = [v for v in vids if v not in already_collected]

    print(f"[info] total de vídeos na fonte:       {len(vids)}")
    print(f"[info] já coletados anteriormente:     {len(already_collected)}")
    print(f"[info] pendentes nesta execução:       {len(pending)}")

    if not pending:
        print("[info] Nenhum vídeo novo para coletar comentários.")
        return

    all_rows: list[dict] = []

    for vid in tqdm(pending, desc="Coletando comentários"):
        try:
            threads = fetch_comments(yt, vid, MAX_PER_VIDEO)
            for t in threads:
                all_rows.extend(flatten(vid, t))
        except HttpError as e:
            print(f"[warn] {vid} sem comentários ou restrito: {e}")
        except Exception as e:
            print(f"[warn] {vid} erro inesperado: {e}")

    if not all_rows:
        print("Nenhum comentário novo coletado.")
        return

    new_df = pd.DataFrame(all_rows)

    out_path = Path("outputs/comments.csv")
    if out_path.exists() and already_collected:
        try:
            old_df = pd.read_csv(out_path)
            new_df = pd.concat([old_df, new_df], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["commentId"], keep="last")
        except Exception as e:
            print(f"[warn] não consegui mesclar com comments.csv existente: {e}")

    new_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Salvo comments.csv com {len(new_df)} linhas no total")


if __name__ == "__main__":
    main()