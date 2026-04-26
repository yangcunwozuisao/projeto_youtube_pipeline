"""
exports_bi.py — Exporta tabelas de BI a partir do dataset enriquecido.

Correção: a versão anterior sobrescrevia dataset_brands.csv com
bi_fato_videos.csv (que tem menos colunas), corrompendo dados de marca
para execuções posteriores. Agora apenas bi_fato_videos.csv é gerado
aqui; dataset_brands.csv é responsabilidade exclusiva de brand_entity_extract.py.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:

    # ── Fonte base ────────────────────────────────────────────────────────────
    for candidate in [
        "outputs/dataset_enriched.csv",
        "outputs/dataset_topics.csv",
        "outputs/dataset_nlp.csv",
    ]:
        if Path(candidate).exists():
            base = candidate
            break
    else:
        raise FileNotFoundError(
            "Nenhuma base encontrada para exportar BI. "
            "Execute os steps de NLP/topics/comments primeiro."
        )

    df = pd.read_csv(base)

    df["viewCount"]  = pd.to_numeric(df.get("viewCount"),  errors="coerce")
    df["sent_value"] = pd.to_numeric(df.get("sent_value"), errors="coerce")
    if "topc_value" in df.columns:
        df["topc_value"] = pd.to_numeric(df.get("topc_value"), errors="coerce")

    # ── Merge com marcas (se disponível) ─────────────────────────────────────
    brands_path = Path("outputs/dataset_brands.csv")
    if brands_path.exists():
        brands = pd.read_csv(brands_path)
        keep_brand_cols = [
            c for c in [
                "videoId", "brand_primary", "brand_series", "model_hint",
                "brands_text", "brand_source", "brand_conf",
            ]
            if c in brands.columns
        ]
        if keep_brand_cols:
            df = df.merge(
                brands[keep_brand_cols].drop_duplicates("videoId"),
                on="videoId",
                how="left",
            )

    # ── Agregado por canal ────────────────────────────────────────────────────
    agg_channel_dict: dict = {
        "videos":      ("videoId",    "nunique"),
        "views_total": ("viewCount",  "sum"),
        "sent_medio":  ("sent_value", "mean"),
    }
    if "topc_value" in df.columns:
        agg_channel_dict["sent_pub_medio"] = ("topc_value", "mean")

    agg_channel = (
        df.groupby("channelTitle", dropna=False)
        .agg(**{k: v for k, v in agg_channel_dict.items() if v[0] in df.columns})
        .reset_index()
        .sort_values("views_total", ascending=False)
    )
    agg_channel.to_csv("outputs/bi_agg_channel.csv", index=False, encoding="utf-8-sig")

    # ── Agregado por marca ────────────────────────────────────────────────────
    if "brand_primary" in df.columns:
        agg_brand_dict: dict = {
            "videos":      ("videoId",    "nunique"),
            "views_total": ("viewCount",  "sum"),
            "sent_medio":  ("sent_value", "mean"),
        }
        if "topc_value" in df.columns:
            agg_brand_dict["sent_pub_medio"] = ("topc_value", "mean")

        agg_brand = (
            df.groupby("brand_primary", dropna=False)
            .agg(**{k: v for k, v in agg_brand_dict.items() if v[0] in df.columns})
            .reset_index()
            .sort_values("views_total", ascending=False)
        )
        agg_brand.to_csv("outputs/bi_agg_brand.csv", index=False, encoding="utf-8-sig")

    # ── Tabela fato para BI ───────────────────────────────────────────────────
    cols_keep = [
        c for c in [
            "videoId", "title", "channelTitle", "publishedAt",
            "viewCount", "language", "keywords",
            "sent_label", "sent_value",
            "topic_id", "topic_repr",
            "brand_primary", "brand_series", "model_hint",
            "brands_text", "brand_source", "brand_conf",
            "top_comment", "topc_likes",
            "topc_label", "topc_conf", "topc_value",
        ]
        if c in df.columns
    ]

    df[cols_keep].to_csv("outputs/bi_fato_videos.csv", index=False, encoding="utf-8-sig")

    # NOTA: dataset_brands.csv NÃO é sobrescrito aqui.
    # Ele é gerado exclusivamente por brand_entity_extract.py e contém
    # colunas extras (brand_conf, brand_source, etc.) que não estão em bi_fato_videos.

    salvos = ["bi_agg_channel.csv", "bi_fato_videos.csv"]
    if "brand_primary" in df.columns:
        salvos.append("bi_agg_brand.csv")

    print("Exports salvos: " + ", ".join(salvos))


if __name__ == "__main__":
    main()