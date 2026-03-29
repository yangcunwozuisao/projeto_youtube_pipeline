import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

API_KEY = "AIzaSyBjHKoD9uHaeNVIcU33CBVXw02x9xf7evU"
QUERY = "unboxing"
PUBLISHED_AFTER = "2015-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2026-12-31T23:59:59Z"
REGION_CODE = "BR"
LANG_HINT = "pt"
MAX_TOTAL = 100  # qtd de video


def build_client():
    return build(
        "youtube",
        "v3",
        developerKey=API_KEY,
        cache_discovery=False
    )


def search_all(youtube):
    items, next_page = [], None

    with tqdm(total=MAX_TOTAL, desc="search") as bar:
        while len(items) < MAX_TOTAL:
            remaining = MAX_TOTAL - len(items)

            resp = youtube.search().list(
                part="id,snippet",
                q=QUERY,
                type="video",
                maxResults=min(50, remaining),
                order="relevance",
                relevanceLanguage=LANG_HINT,
                regionCode=REGION_CODE,
                publishedAfter=PUBLISHED_AFTER,
                publishedBefore=PUBLISHED_BEFORE,
                pageToken=next_page
            ).execute()

            batch = [
                it for it in resp.get("items", [])
                if it.get("id", {}).get("videoId")
            ]

            items.extend(batch)
            bar.update(len(batch))

            next_page = resp.get("nextPageToken")
            if not next_page:
                break

            time.sleep(0.2)

    return items[:MAX_TOTAL]


def enrich_videos(youtube, video_ids):
    rows = []

    for i in tqdm(range(0, len(video_ids), 50), desc="enrich"):
        chunk = video_ids[i:i+50]

        resp = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk)
        ).execute()

        for it in resp.get("items", []):
            snip = it.get("snippet", {})
            stats = it.get("statistics", {})
            cd = it.get("contentDetails", {})

            rows.append({
                "videoId": it.get("id"),
                "title": snip.get("title"),
                "description": snip.get("description"),
                "channelTitle": snip.get("channelTitle"),
                "publishedAt": snip.get("publishedAt"),
                "duration": cd.get("duration"),
                "viewCount": stats.get("viewCount"),
                "likeCount": stats.get("likeCount"),
                "commentCount": stats.get("commentCount"),
            })

        time.sleep(0.1)

    return pd.DataFrame(rows)


def main():
    if not API_KEY or API_KEY == "COLOQUE_SUA_NOVA_CHAVE_AQUI":
        raise RuntimeError("Defina uma API key válida em API_KEY.")

    try:
        yt = build_client()
        items = search_all(yt)

        if not items:
            print("Nenhum vídeo encontrado.")
            return

        ids = [it["id"]["videoId"] for it in items]
        df = enrich_videos(yt, ids)

        df.to_csv("outputs/videos.csv", index=False, encoding="utf-8-sig")
        print(f" Salvo {len(df)} vídeos em videos.csv")

    except HttpError as e:
        print(f"Erro da API do YouTube: {e}")
    except Exception as e:
        print(f"Erro geral: {e}")


if __name__ == "__main__":
    main()