from pathlib import Path
import pandas as pd

def main():

    SRC = (
        "outputs/dataset_enriched.csv"
        if Path("outputs/dataset_enriched.csv").exists()
        else (
            "outputs/dataset_topics.csv"
            if Path("outputs/dataset_topics.csv").exists()
            else "outputs/dataset_nlp.csv"
        )
    )

    df = pd.read_csv(SRC)

    df["viewCount"] = pd.to_numeric(df.get("viewCount"), errors="coerce")
    df["sent_value"] = pd.to_numeric(df.get("sent_value"), errors="coerce")

    if "topc_value" in df.columns:
        df["topc_value"] = pd.to_numeric(df.get("topc_value"), errors="coerce")

    agg = (
        df.groupby("channelTitle", dropna=False)
        .agg(
            videos=("videoId", "nunique"),
            views_total=("viewCount", "sum"),
            views_med=("viewCount", "median"),
            sent_medio=("sent_value", "mean"),
            **(
                {"sent_pub_medio": ("topc_value", "mean")}
                if "topc_value" in df.columns
                else {}
            )
        )
        .reset_index()
        .sort_values("views_total", ascending=False)
    )

    agg.to_csv(
        "outputs/agg_channel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"salvo agg_channel.csv com {len(agg)} linhas")


if __name__ == "__main__":
    main()