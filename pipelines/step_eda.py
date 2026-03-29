import pandas as pd
import re


def iso8601_duration_to_seconds(s):
    if pd.isna(s):
        return None
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", s)
    if not m:
        return None
    h = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    ss = int(m.group(3) or 0)
    return h * 3600 + mm * 60 + ss


def main():

    df = pd.read_csv("outputs/videos.csv")

    df["duration_sec"] = df["duration"].apply(iso8601_duration_to_seconds)
    df["published_date"] = pd.to_datetime(df["publishedAt"], errors="coerce").dt.date
    df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce")

    print("linhas:", len(df))

    print("\nTop 10 por visualizações:")
    print(
        df[["title", "channelTitle", "viewCount"]]
        .sort_values("viewCount", ascending=False)
        .head(10)
    )

    print("\nCanais mais frequentes:")
    print(df["channelTitle"].value_counts().head(10))

    df.to_csv("outputs/videos_clean.csv", index=False, encoding="utf-8-sig")

    print("\n Salvo videos_clean.csv com duration_sec e published_date")


if __name__ == "__main__":
    main()
