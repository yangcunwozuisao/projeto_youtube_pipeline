import os
import time
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tqdm import tqdm

from tools.utils_incremental import get_last_published_at


# Configuração
API_KEY = os.getenv("YT_API_KEY", "")

PUBLISHED_AFTER_DEFAULT = "2015-01-01T00:00:00Z"
PUBLISHED_BEFORE = "2026-12-31T23:59:59Z"

# Analisando o número de dias: Evitando uma situação em que você dificilmente conseguirá detectar algo "após o último ponto de tempo".
LOOKBACK_DAYS = 30

# ignorar palavras q pode ser acessorio
EXCLUDE_PATTERNS = [
    r"\bcase\b",
    r"\bcover\b",
    r"\bwallet\b",
    r"\bphone case\b",
    r"\bretro\b",
    r"\blot\b",
    r"\barchive\b",
]

MARKETS = [
    {
        "name": "BR",
        "regionCode": "BR",
        "language": "pt",
        "max_total": 120,
        "queries": [
            "unboxing iphone brasil",
            "review iphone brasil",
            "review samsung brasil",
            "xiaomi review brasil",
            "motorola review brasil",
            "comparativo celulares",
            "melhor celular custo beneficio",
            "smartphone brasil review",
            "celular review brasil",
        ],
    },
    {
        "name": "US",
        "regionCode": "US",
        "language": "en",
        "max_total": 120,
        "queries": [
            "iphone unboxing",
            "iphone review",
            "samsung review",
            "pixel review",
            "xiaomi review",
            "motorola review",
            "best smartphone review",
            "smartphone comparison",
            "phone camera comparison",
        ],
    },
    {
        "name": "GLOBAL",
        "regionCode": None,
        "language": None,
        "max_total": 60,
        "queries": [
            "iphone unboxing",
            "samsung unboxing",
            "smartphone review",
            "android phone review",
            "smartphone comparison",
        ],
    },
]


def build_client():
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def compute_published_after(last_date: str | None, fallback: str, lookback_days: int = 30) -> str:
    """
    Em vez de usar incremental estrito, volta alguns dias no tempo
    para aumentar a cobertura e evitar amostra muito pequena.
    """
    if not last_date:
        return fallback

    try:
        dt = datetime.fromisoformat(last_date.replace("Z", "+00:00"))
        dt = dt - timedelta(days=lookback_days)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        print(f"[warn] não consegui interpretar last_date={last_date!r}: {e}")
        return fallback


def looks_relevant_text(title: str | None, description: str | None = None) -> bool:
    text = f"{title or ''} {description or ''}".lower()

    for pattern in EXCLUDE_PATTERNS:
        if re.search(pattern, text, flags=re.I):
            return False

    return True


# Busca

def search_one_query(
    youtube,
    query: str,
    max_items: int,
    published_after: str,
    region_code: str | None = None,
    language: str | None = None,
):
    items = []
    next_page = None

    while len(items) < max_items:
        remaining = max_items - len(items)

        params = {
            "part": "id,snippet",
            "q": query,
            "type": "video",
            "maxResults": min(50, remaining),
            "order": "date",
            "publishedAfter": published_after,
            "publishedBefore": PUBLISHED_BEFORE,
            "pageToken": next_page,
        }

        if region_code:
            params["regionCode"] = region_code
        if language:
            params["relevanceLanguage"] = language

        resp = youtube.search().list(**params).execute()

        batch = [
            it for it in resp.get("items", [])
            if it.get("id", {}).get("videoId")
        ]
        if not batch:
            break

        for it in batch:
            it["_search_query"] = query
            it["_market"] = region_code or "GLOBAL"
            it["_language_hint"] = language or "any"

        items.extend(batch)

        next_page = resp.get("nextPageToken")
        if not next_page:
            break

        time.sleep(0.2)

    return items


def search_market(youtube, market_cfg: dict, published_after: str):
    queries = market_cfg["queries"]
    max_total = int(market_cfg["max_total"])
    region_code = market_cfg.get("regionCode")
    language = market_cfg.get("language")

    per_query = max(12, max_total // max(len(queries), 1) + 8)
    collected = []

    print(
        f"[INFO] Mercado {market_cfg['name']} | "
        f"region={region_code} | lang={language} | max_total={max_total}"
    )

    for query in tqdm(queries, desc=f"queries {market_cfg['name']}"):
        try:
            batch = search_one_query(
                youtube=youtube,
                query=query,
                max_items=per_query,
                published_after=published_after,
                region_code=region_code,
                language=language,
            )
            collected.extend(batch)
        except HttpError as e:
            print(f"[warn] erro na query '{query}' ({market_cfg['name']}): {e}")
        except Exception as e:
            print(f"[warn] erro geral na query '{query}' ({market_cfg['name']}): {e}")

    # dedup por videoId dentro do mercado
    dedup = {}
    for item in collected:
        vid = item.get("id", {}).get("videoId")
        if vid and vid not in dedup:
            dedup[vid] = item

    items = sorted(
        dedup.values(),
        key=lambda x: x.get("snippet", {}).get("publishedAt", ""),
        reverse=True,
    )

    return items[:max_total]


def search_all(youtube):
    last_date = get_last_published_at()
    published_after = compute_published_after(
        last_date=last_date,
        fallback=PUBLISHED_AFTER_DEFAULT,
        lookback_days=LOOKBACK_DAYS
    )

    if last_date:
        print(f"[INFO] Último vídeo no banco: {last_date}")
        print(f"[INFO] Incremental com janela de retorno: coletando após {published_after}")
    else:
        print(f"[INFO] Coleta completa → coletando vídeos após {PUBLISHED_AFTER_DEFAULT}")

    all_items = []
    for market in MARKETS:
        market_items = search_market(youtube, market, published_after)
        all_items.extend(market_items)

    # dedup global por videoId
    dedup = {}
    for item in all_items:
        vid = item.get("id", {}).get("videoId")
        if vid and vid not in dedup:
            dedup[vid] = item

    items = sorted(
        dedup.values(),
        key=lambda x: x.get("snippet", {}).get("publishedAt", ""),
        reverse=True,
    )

    print(f"[INFO] Total bruto após merge global: {len(items)}")
    return items


# Enriquecimento de vídeos

def enrich_videos(youtube, items: list[dict]) -> pd.DataFrame:
    rows = []

    item_map = {}
    for it in items:
        vid = it.get("id", {}).get("videoId")
        if vid:
            item_map[vid] = it

    video_ids = list(item_map.keys())

    for i in tqdm(range(0, len(video_ids), 50), desc="enrich videos"):
        chunk = video_ids[i : i + 50]

        resp = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(chunk),
        ).execute()

        for it in resp.get("items", []):
            vid = it.get("id")
            raw = item_map.get(vid, {})

            snip = it.get("snippet", {})
            stats = it.get("statistics", {})
            cd = it.get("contentDetails", {})

            rows.append({
                "videoId": vid,
                "channelId": snip.get("channelId"),
                "title": snip.get("title"),
                "description": snip.get("description"),
                "channelTitle": snip.get("channelTitle"),
                "publishedAt": snip.get("publishedAt"),
                "duration": cd.get("duration"),

                "market": raw.get("_market"),
                "search_query": raw.get("_search_query"),
                "language_hint": raw.get("_language_hint"),

                "viewCount": stats.get("viewCount"),
                "likeCount": stats.get("likeCount"),
                "commentCount": stats.get("commentCount"),
                "dislikeCount": None,
            })

        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["videoId"])

        # filtrar os que nao importa
        before = len(df)
        df = df[
            df.apply(
                lambda r: looks_relevant_text(r.get("title"), r.get("description")),
                axis=1
            )
        ].copy()
        after = len(df)
        print(f"[INFO] Filtragem por relevância textual: {before} -> {after}")

    return df


# Enriquecimento de canais

def enrich_channels(youtube, channel_ids: list[str]) -> pd.DataFrame:
    rows = []
    unique_ids = list(set(channel_ids))

    for i in tqdm(range(0, len(unique_ids), 50), desc="enrich channels"):
        chunk = unique_ids[i : i + 50]

        resp = youtube.channels().list(
            part="snippet,statistics",
            id=",".join(chunk),
        ).execute()

        for it in resp.get("items", []):
            stats = it.get("statistics", {})
            snip = it.get("snippet", {})

            rows.append({
                "channelId": it.get("id"),
                "channelDescription": snip.get("description"),
                "channelCountry": snip.get("country"),
                "subscriberCount": stats.get("subscriberCount"),
                "channelVideoCount": stats.get("videoCount"),
                "channelViewCount": stats.get("viewCount"),
                "hiddenSubscriberCount": stats.get("hiddenSubscriberCount", False),
            })

        time.sleep(0.1)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["channelId"])

    return df


# Main

def main():
    if not API_KEY:
        raise RuntimeError(
            "API key não encontrada.\n"
            "Defina a variável de ambiente YT_API_KEY ou crie um arquivo .env."
        )

    try:
        yt = build_client()

        # 1) Busca lista de vídeos
        items = search_all(yt)
        if not items:
            print("Nenhum vídeo novo encontrado.")
            return

        # 2) Enriquece vídeos
        df_videos = enrich_videos(yt, items)

        if df_videos.empty:
            print("Nenhum vídeo enriquecido com sucesso.")
            return

        # 3) Enriquece canais
        channel_ids = df_videos["channelId"].dropna().unique().tolist()
        df_channels = enrich_channels(yt, channel_ids)

        # 4) Join vídeos + canais
        df = df_videos.merge(df_channels, on="channelId", how="left")

        # 5) Numéricos
        numeric_cols = [
            "viewCount",
            "likeCount",
            "commentCount",
            "subscriberCount",
            "channelVideoCount",
            "channelViewCount",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 6) Ordena por data mais recente
        df = df.sort_values("publishedAt", ascending=False)

        # 7) Salva
        df.to_csv("outputs/videos.csv", index=False, encoding="utf-8-sig")

        print(f"Salvo {len(df)} vídeos em outputs/videos.csv")
        print(f"Canais únicos: {df['channelId'].nunique()}")
        print(f"Mercados: {df['market'].value_counts(dropna=False).to_dict() if 'market' in df.columns else {}}")
        print(f"Com subscriberCount: {df['subscriberCount'].notna().sum()}")
        print(f"Com inscritos ocultos: {df['hiddenSubscriberCount'].sum() if 'hiddenSubscriberCount' in df.columns else 0}")

    except HttpError as e:
        print(f"Erro da API do YouTube: {e}")
    except Exception as e:
        print(f"Erro geral: {e}")


if __name__ == "__main__":
    main()