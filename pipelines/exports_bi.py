# file: exports_bi.py

from pathlib import Path
import pandas as pd


def main():

    base = (
        "outputs/dataset_enriched.csv"
        if Path("outputs/dataset_enriched.csv").exists()
        else (
            "outputs/dataset_topics.csv"
            if Path("outputs/dataset_topics.csv").exists()
            else "outputs/dataset_nlp.csv"
        )
    )

    df = pd.read_csv(base)

    df["viewCount"] = pd.to_numeric(df.get("viewCount"), errors="coerce")
    df["sent_value"] = pd.to_numeric(df.get("sent_value"), errors="coerce")

    if "topc_value" in df.columns:
        df["topc_value"] = pd.to_numeric(df.get("topc_value"), errors="coerce")

    brands = (
        pd.read_csv("outputs/dataset_brands.csv")
        if Path("outputs/dataset_brands.csv").exists()
        else None
    )

    if brands is not None:
        keep_brand_cols = [c for c in ["videoId", "brand_primary", "model_hint"] if c in brands.columns]

        if keep_brand_cols:
            df = df.merge(
                brands[keep_brand_cols].drop_duplicates("videoId"),
                on="videoId",
                how="left"
            )

    # agregados por canal
    agg_channel = (
        df.groupby("channelTitle", dropna=False)
        .agg(
            videos=("videoId", "nunique"),
            views_total=("viewCount", "sum"),
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

    agg_channel.to_csv(
        "outputs/bi_agg_channel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # agregados por marca
    if "brand_primary" in df.columns:
        agg_brand = (
            df.groupby("brand_primary", dropna=False)
            .agg(
                videos=("videoId", "nunique"),
                views_total=("viewCount", "sum"),
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

        agg_brand.to_csv(
            "outputs/bi_agg_brand.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # tabela fato para BI
    cols_keep = [
        c for c in [
            "videoId",
            "title",
            "channelTitle",
            "publishedAt",
            "viewCount",
            "language",
            "keywords",
            "sent_label",
            "sent_value",
            "topic_id",
            "topic_repr",
            "brand_primary",
            "model_hint",
            "top_comment",
            "topc_likes",
            "topc_label",
            "topc_conf",
            "topc_value"
        ] if c in df.columns
    ]

    df[cols_keep].to_csv(
        "outputs/bi_fato_videos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "exports salvos: bi_agg_channel.csv, bi_agg_brand.csv (se marcas), bi_fato_videos.csv"
    )


if __name__ == "__main__":
    main()