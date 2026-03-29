# file: exports_bi.py

from pathlib import Path
import pandas as pd


def main():

    # carrega bases (usa as que existirem)
    base = (
        "outputs/dataset_topics.csv"
        if Path("outputs/dataset_topics.csv").exists()
        else "outputs/dataset_nlp.csv"
    )

    df = pd.read_csv(base)

    brands = (
        pd.read_csv("outputs/dataset_brands.csv")
        if Path("outputs/dataset_brands.csv").exists()
        else None
    )

    # merge opcional com brands
    if brands is not None:

        df = df.merge(
            brands[["videoId", "brand_primary", "model_hint"]],
            on="videoId",
            how="left"
        )

    # agregados por canal
    agg_channel = (
        df.groupby("channelTitle", dropna=False)
        .agg(
            videos=("videoId", "nunique"),
            views_total=("viewCount", "sum"),
            sent_medio=("sent_value", "mean")
        )
        .reset_index()
        .sort_values("views_total", ascending=False)
    )

    agg_channel.to_csv(
        "outputs/bi_agg_channel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # agregados por marca (se existir)
    if "brand" in df.columns:

        agg_brand = (
            df.groupby("brand_primary", dropna=False)   
            .agg(
                videos=("videoId", "nunique"),
                views_total=("viewCount", "sum"),
                sent_medio=("sent_value", "mean")
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
            "brand",
            "model_hint"
        ] if c in df.columns
    ]

    df[cols_keep].to_csv(
        "outputs/bi_fato_videos.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        " exports salvos: bi_agg_channel.csv, bi_agg_brand.csv (se marcas), bi_fato_videos.csv"
    )


if __name__ == "__main__":
    main()
