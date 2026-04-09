import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from tools.utils_incremental import get_last_published_at

API_KEY = "AIzaSyBjHKoD9uHaeNVIcU33CBVXw02x9xf7evU"
QUERIES = [
    "iphone unboxing",
    "samsung unboxing",
    "xiaomi unboxing",
    "motorola unboxing",
    "realme unboxing",
    "oppo unboxing",
    "vivo unboxing",
    "oneplus unboxing",
    "huawei unboxing",
    "google pixel unboxing",
    "asus rog phone unboxing",
    "nothing phone unboxing",
    "infinix unboxing",
    "tecno unboxing",
    "smartphone review",
    "celular review",
    "comparativo smartphones"
]
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


def search_one_query(youtube, query, max_items, published_after):
    items = []
    next_page = None

    while len(items) < max_items:
        remaining = max_items - len(items)

        resp = youtube.search().list(
            part="id,snippet",
            q=query,
            type="video",
            maxResults=min(50, remaining),
            order="date",
            relevanceLanguage=LANG_HINT,
            regionCode=REGION_CODE,
            publishedAfter=published_after,
            publishedBefore=PUBLISHED_BEFORE,
            pageToken=next_page
        ).execute()

        batch = [
            it for it in resp.get("items", [])
            if it.get("id", {}).get("videoId")
        ]

        if not batch:
            break

        items.extend(batch)

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

        time.sleep(0.2)

    return items


def search_all(youtube):
    all_items = []

    last_date = get_last_published_at()
    published_after = last_date if last_date else PUBLISHED_AFTER

    if last_date:
        print(f"[INFO] Modo incremental ativado -> coletando vídeos após {last_date}")
    else:
        print(f"[INFO] Coleta completa ativada -> coletando vídeos após {PUBLISHED_AFTER}")

    per_query = max(10, MAX_TOTAL // len(QUERIES) + 5)

    for query in tqdm(QUERIES, desc="queries"):
        try:
            batch = search_one_query(
                youtube=youtube,
                query=query,
                max_items=per_query,
                published_after=published_after
            )
            all_items.extend(batch)
        except HttpError as e:
            print(f"[warn] erro na query '{query}': {e}")
        except Exception as e:
            print(f"[warn] erro geral na query '{query}': {e}")

    dedup = {}
    for item in all_items:
        vid = item.get("id", {}).get("videoId")
        if vid and vid not in dedup:
            dedup[vid] = item

    items = list(dedup.values())

    items = sorted(
        items,
        key=lambda x: x.get("snippet", {}).get("publishedAt", ""),
        reverse=True
    )

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

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["videoId"])

    return df


def main():
    if not API_KEY or API_KEY == "COLOQUE_SUA_NOVA_CHAVE_AQUI":
        raise RuntimeError("Defina uma API key válida em API_KEY.")

    try:
        yt = build_client()
        items = search_all(yt)

        if not items:
            print("Nenhum vídeo novo encontrado.")
            return

        ids = [it["id"]["videoId"] for it in items]
        df = enrich_videos(yt, ids)

        df.to_csv("outputs/videos.csv", index=False, encoding="utf-8-sig")
        print(f"Salvo {len(df)} vídeos em outputs/videos.csv")

    except HttpError as e:
        print(f"Erro da API do YouTube: {e}")
    except Exception as e:
        print(f"Erro geral: {e}")


if __name__ == "__main__":
    main()