"""
dashboard.py — Streamlit dashboard para análise dos dados do pipeline.
"""

from __future__ import annotations

import os
from collections import Counter as _Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


st.set_page_config(page_title="YouTube NLP Dashboard", layout="wide")

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "youtube_db")

def safe_mean(series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else 0.0


def fmt_big(n) -> str:
    if pd.isna(n):
        return "—"

    n = float(n)

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"

    return f"{n:.0f}"


def clean_text_for_plotly(x):
    try:
        if x is None:
            return "unknown"
        if not isinstance(x, (list, dict, tuple, set)) and pd.isna(x):
            return "unknown"
    except Exception:
        pass

    text = str(x)

    text = text.replace("\ufffd", "")

    text = "".join(
        ch for ch in text
        if ch == "\n" or ch == "\t" or ord(ch) >= 32
    )

    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = text.strip()

    return text if text else "unknown"


def sum_keep_nan(series):
    s = pd.to_numeric(series, errors="coerce")
    return s.sum(min_count=1)


def fmt_chart_value(n):
    if pd.isna(n):
        return "sem dado"

    n = float(n)

    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"

    return f"{n:.0f}"

# MongoDB

@st.cache_resource
def get_mongo_client():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")
    return client


def get_db():
    return get_mongo_client()[MONGO_DB]


# Carga de dados

@st.cache_data(ttl=120)
def load_data() -> tuple[pd.DataFrame, str]:
    db = get_db()

    videos = pd.DataFrame(list(db["videos"].find({}, {"_id": 0})))
    nlp = pd.DataFrame(list(db["nlp"].find({}, {"_id": 0})))
    topics = pd.DataFrame(list(db["topics"].find({}, {"_id": 0})))
    brands = pd.DataFrame(list(db["brands"].find({}, {"_id": 0})))

    if videos.empty and brands.empty:
        raise ValueError("Coleções 'videos' e 'brands' vazias. Execute o pipeline primeiro.")

    if not videos.empty:
        df = videos.copy()
        source = f"MongoDB › videos ({len(videos)} docs)"

        for sub, name in [
            (nlp, "nlp"),
            (topics, "topics"),
            (brands, "brands"),
        ]:
            if not sub.empty and "videoId" in sub.columns:
                extra_cols = [
                    c for c in sub.columns
                    if c not in df.columns or c == "videoId"
                ]

                df = df.merge(
                    sub[extra_cols].drop_duplicates("videoId"),
                    on="videoId",
                    how="left",
                )

                source += f" + {name}"

    else:
        df = brands.copy()
        source = f"MongoDB › brands ({len(brands)} docs)"

    num_cols = [
        "viewCount",
        "likeCount",
        "commentCount",
        "dislikeCount",
        "subscriberCount",
        "channelVideoCount",
        "channelViewCount",
        "sent_value",
        "topc_likes",
        "topc_conf",
        "topc_value",
        "brand_conf",
    ]

    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    str_cols = [
        "brand_primary",
        "sent_label",
        "topc_label",
        "channelTitle",
        "model_hint",
    ]

    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)

    if {"likeCount", "viewCount"}.issubset(df.columns):
        df["like_rate"] = (
            df["likeCount"] / df["viewCount"].replace(0, np.nan) * 100
        ).round(4)

    if {"commentCount", "viewCount"}.issubset(df.columns):
        df["comment_rate"] = (
            df["commentCount"] / df["viewCount"].replace(0, np.nan) * 100
        ).round(4)

    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        df[col] = df[col].apply(clean_text_for_plotly)

    return df, source


@st.cache_data(ttl=120)
def load_collection_stats() -> dict:
    db = get_db()
    cols = ["videos", "transcripts", "nlp", "topics", "brands", "comments"]
    return {c: db[c].count_documents({}) for c in cols}


@st.cache_data(ttl=120)
def load_spam_data() -> pd.DataFrame | None:
    db = get_db()

    count = db["comments"].count_documents({"spam_label": {"$exists": True}})

    if count == 0:
        return None

    records = list(
        db["comments"].find(
            {"spam_label": {"$exists": True}},
            {"_id": 0},
        )
    )

    df = pd.DataFrame(records)

    df["spam_score"] = pd.to_numeric(df.get("spam_score"), errors="coerce").fillna(0)
    df["likeCount"] = pd.to_numeric(df.get("likeCount"), errors="coerce").fillna(0)

    if "spam_signals" in df.columns:
        df["spam_signals"] = df["spam_signals"].fillna("").astype(str)

    text_cols = df.select_dtypes(include=["object"]).columns

    for col in text_cols:
        if col != "spam_signals":
            df[col] = df[col].apply(clean_text_for_plotly)

    return df

def main() -> None:
    st.title("YouTube NLP Dashboard")

    try:
        get_mongo_client()
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        st.error(
            f"Não foi possível conectar ao MongoDB em `{MONGO_URI}`.\n\n"
            f"Verifique se o serviço está rodando ou defina a variável de ambiente "
            f"`MONGO_URI`.\n\nDetalhe: {e}"
        )
        st.stop()

    try:
        df, source = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()

    has_comments = "topc_value" in df.columns and df["topc_value"].notna().any()
    has_subscribers = "subscriberCount" in df.columns and df["subscriberCount"].notna().any()

    # Sidebar
    with st.sidebar:
        st.header("Filtros")

        selected_brands = None
        if "brand_primary" in df.columns:
            brands = sorted(df["brand_primary"].dropna().unique().tolist())
            selected_brands = st.multiselect("Marca", brands, default=brands)

        selected_sent = None
        if "sent_label" in df.columns:
            sent_labels = sorted(df["sent_label"].dropna().unique().tolist())
            selected_sent = st.multiselect(
                "Sentimento do vídeo",
                sent_labels,
                default=sent_labels,
            )

        selected_topc = None
        if "topc_label" in df.columns:
            comment_labels = sorted(df["topc_label"].dropna().astype(str).unique().tolist())
            selected_topc = st.multiselect(
                "Sentimento do comentário",
                comment_labels,
                default=comment_labels,
            )

        selected_channels = None
        if "channelTitle" in df.columns:
            channels = sorted(df["channelTitle"].dropna().unique().tolist())
            selected_channels = st.multiselect("Canal", channels, default=channels)

        st.divider()
        st.subheader("MongoDB — coleções")

        try:
            for col, count in load_collection_stats().items():
                st.metric(col, count)
        except Exception:
            st.warning("Não foi possível carregar estatísticas.")

        if st.button("Recarregar dados"):
            st.cache_data.clear()
            st.rerun()

    # Aplicar filtros
    df_f = df.copy()

    if selected_brands and "brand_primary" in df_f.columns:
        df_f = df_f[df_f["brand_primary"].isin(selected_brands)]

    if selected_sent and "sent_label" in df_f.columns:
        df_f = df_f[df_f["sent_label"].isin(selected_sent)]

    if selected_topc and "topc_label" in df_f.columns:
        df_f = df_f[df_f["topc_label"].isin(selected_topc)]

    if selected_channels and "channelTitle" in df_f.columns:
        df_f = df_f[df_f["channelTitle"].isin(selected_channels)]

    ci, cw = st.columns([3, 1])

    with ci:
        st.success(f"Fonte: **{source}** — {len(df_f)} vídeos (de {len(df)})")

    with cw:
        if not has_comments:
            st.warning("Sem topc_*. Execute steps 7 e 8.")

    #1 — MÉTRICAS GERAIS

    st.header("Métricas gerais")

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        st.metric("Vídeos", len(df_f))

    with m2:
        st.metric(
            "Canais únicos",
            df_f["channelTitle"].nunique() if "channelTitle" in df_f.columns else "—",
        )

    with m3:
        st.metric("Views (média)", fmt_big(safe_mean(df_f.get("viewCount"))))

    with m4:
        st.metric("Likes (média)", fmt_big(safe_mean(df_f.get("likeCount"))))

    with m5:
        st.metric(
            "Inscritos (média canal)",
            fmt_big(safe_mean(df_f.get("subscriberCount"))) if has_subscribers else "—",
        )

    with m6:
        st.metric(
            "Like rate (%)",
            f"{safe_mean(df_f.get('like_rate', pd.Series(dtype=float))):.2f}",
        )

    st.divider()

    # 2 — BI DE CORRELAÇÃO

    st.header("BI — Correlação de métricas")

    corr_candidates = [
        c for c in [
            "viewCount",
            "likeCount",
            "commentCount",
            "subscriberCount",
            "channelViewCount",
            "like_rate",
            "comment_rate",
            "sent_value",
            "topc_value",
        ]
        if c in df_f.columns
    ]

    corr_labels = {
        "viewCount": "Views",
        "likeCount": "Likes",
        "commentCount": "Comentários",
        "subscriberCount": "Inscritos canal",
        "channelViewCount": "Views canal (total)",
        "like_rate": "Like rate (%)",
        "comment_rate": "Comment rate (%)",
        "sent_value": "Sentimento vídeo",
        "topc_value": "Sentimento comentário",
    }

    if len(corr_candidates) >= 2:
        corr_df = df_f[corr_candidates].apply(pd.to_numeric, errors="coerce")
        corr_matrix = corr_df.corr(method="pearson")
        labels = [corr_labels.get(c, c) for c in corr_candidates]

        fig_heat = go.Figure(
            go.Heatmap(
                z=corr_matrix.values,
                x=labels,
                y=labels,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
            )
        )

        fig_heat.update_layout(
            title="Correlação de Pearson entre métricas",
            height=420,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Dados insuficientes para matriz de correlação.")

    st.divider()

    # Scatter plots

    col_a, col_b = st.columns(2)

    with col_a:
        if {"viewCount", "likeCount"}.issubset(df_f.columns):
            plot_df = df_f.copy()
            plot_df = plot_df[
                (plot_df["viewCount"] > 0) &
                (plot_df["likeCount"] > 0)
            ]

            hover = [c for c in ["title", "channelTitle", "brand_primary"] if c in plot_df.columns]
            color = "brand_primary" if "brand_primary" in plot_df.columns else None

            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="viewCount",
                    y="likeCount",
                    color=color,
                    hover_data=hover,
                    log_x=True,
                    log_y=True,
                    labels={
                        "viewCount": "Views (log)",
                        "likeCount": "Likes (log)",
                    },
                    title="Views × Likes  (escala log)",
                )

                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados insuficientes para gráfico Views × Likes.")

    with col_b:
        if has_subscribers and "viewCount" in df_f.columns:
            plot_df = df_f.copy()
            plot_df = plot_df[
                (plot_df["subscriberCount"] > 0) &
                (plot_df["viewCount"] > 0)
            ]

            hover = [c for c in ["channelTitle", "brand_primary"] if c in plot_df.columns]
            color = "brand_primary" if "brand_primary" in plot_df.columns else None

            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="subscriberCount",
                    y="viewCount",
                    color=color,
                    hover_data=hover,
                    log_x=True,
                    log_y=True,
                    labels={
                        "subscriberCount": "Inscritos (log)",
                        "viewCount": "Views (log)",
                    },
                    title="Inscritos do canal × Views  (escala log)",
                )

                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados insuficientes para gráfico Inscritos × Views.")
        else:
            st.info("Dados de inscritos não disponíveis. Rode o pipeline completo.")

    col_c, col_d = st.columns(2)

    with col_c:
        if {"viewCount", "commentCount"}.issubset(df_f.columns):
            plot_df = df_f.copy()
            plot_df = plot_df[
                (plot_df["viewCount"] > 0) &
                (plot_df["commentCount"] > 0)
            ]

            hover = [c for c in ["title", "channelTitle"] if c in plot_df.columns]
            color = "brand_primary" if "brand_primary" in plot_df.columns else None

            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="viewCount",
                    y="commentCount",
                    color=color,
                    hover_data=hover,
                    log_x=True,
                    log_y=True,
                    labels={
                        "viewCount": "Views (log)",
                        "commentCount": "Comentários (log)",
                    },
                    title="Views × Comentários  (escala log)",
                )

                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

    with col_d:
        if has_subscribers and "likeCount" in df_f.columns:
            plot_df = df_f.copy()
            plot_df = plot_df[
                (plot_df["subscriberCount"] > 0) &
                (plot_df["likeCount"] > 0)
            ]

            hover = [c for c in ["channelTitle", "brand_primary"] if c in plot_df.columns]
            color = "brand_primary" if "brand_primary" in plot_df.columns else None

            if not plot_df.empty:
                fig = px.scatter(
                    plot_df,
                    x="subscriberCount",
                    y="likeCount",
                    color=color,
                    hover_data=hover,
                    log_x=True,
                    log_y=True,
                    labels={
                        "subscriberCount": "Inscritos (log)",
                        "likeCount": "Likes (log)",
                    },
                    title="Inscritos do canal × Likes  (escala log)",
                )

                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Ranking de canais

    st.subheader("Ranking de canais")

    if "channelTitle" in df_f.columns:
        agg_dict: dict = {
            "videoId": ("videoId", "nunique"),
            "viewCount": ("viewCount", sum_keep_nan),
            "likeCount": ("likeCount", sum_keep_nan),
            "commentCount": ("commentCount", sum_keep_nan),
            "like_rate": ("like_rate", "mean"),
        }

        if has_subscribers:
            agg_dict["subscriberCount"] = ("subscriberCount", sum_keep_nan)

        ch_agg = (
            df_f.groupby("channelTitle", dropna=False)
            .agg(**{k: v for k, v in agg_dict.items() if v[0] in df_f.columns})
            .reset_index()
            .sort_values("viewCount", ascending=False)
        )

        rename_map = {
            "channelTitle": "Canal",
            "videoId": "Vídeos",
            "viewCount": "Views (total)",
            "likeCount": "Likes (total)",
            "commentCount": "Comentários (total)",
            "like_rate": "Like rate (%) médio",
            "subscriberCount": "Inscritos",
        }

        ch_display = ch_agg.rename(
            columns={k: v for k, v in rename_map.items() if k in ch_agg.columns}
        )

        for col in ["Views (total)", "Likes (total)", "Comentários (total)", "Inscritos"]:
            if col in ch_display.columns:
                ch_display[col] = ch_display[col].apply(
                    lambda x: fmt_big(x) if pd.notna(x) else "—"
                )

        if "Like rate (%) médio" in ch_display.columns:
            ch_display["Like rate (%) médio"] = ch_display["Like rate (%) médio"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—"
            )

        st.dataframe(ch_display, use_container_width=True, hide_index=True)

        top15 = ch_agg.head(15).copy()

        top15["views_text"] = top15["viewCount"].apply(fmt_chart_value)

        if "likeCount" in top15.columns:
            top15["likes_text"] = top15["likeCount"].apply(fmt_chart_value)
        else:
            top15["likes_text"] = "sem coluna"

        if "commentCount" in top15.columns:
            top15["comments_text"] = top15["commentCount"].apply(fmt_chart_value)
        else:
            top15["comments_text"] = "sem coluna"

        color_arg = (
            "likeCount"
            if "likeCount" in top15.columns and top15["likeCount"].notna().any()
            else None
        )

        chart_title = (
            "Top 15 canais por views totais (cor = likes totais)"
            if color_arg
            else "Top 15 canais por views totais"
        )

        fig_ch = px.bar(
            top15,
            x="channelTitle",
            y="viewCount",
            color=color_arg,
            text="views_text",
            color_continuous_scale="Blues" if color_arg else None,
            labels={
                "channelTitle": "Canal",
                "viewCount": "Views",
                "likeCount": "Likes",
            },
            title=chart_title,
            custom_data=[
                "channelTitle",
                "views_text",
                "likes_text",
                "comments_text",
            ],
        )

        fig_ch.update_traces(
            textposition="outside",
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Views: %{customdata[1]}<br>"
                "Likes: %{customdata[2]}<br>"
                "Comentários: %{customdata[3]}"
                "<extra></extra>"
            ),
        )

        fig_ch.update_layout(
            height=430,
            xaxis_tickangle=-35,
            yaxis_title="Views",
            margin=dict(l=20, r=20, t=60, b=80),
        )

        if color_arg:
            fig_ch.update_layout(coloraxis_colorbar_title="Likes")

        st.plotly_chart(fig_ch, use_container_width=True)

    st.divider()

    # Like rate × Sentimento

    st.subheader("Like rate × Sentimento do vídeo")

    if {"like_rate", "sent_value"}.issubset(df_f.columns):
        plot_df = df_f.copy()
        plot_df = plot_df[pd.to_numeric(plot_df["like_rate"], errors="coerce").notna()]

        hover = [c for c in ["title", "channelTitle", "brand_primary", "viewCount"] if c in plot_df.columns]
        color = "brand_primary" if "brand_primary" in plot_df.columns else None

        if not plot_df.empty:
            fig = px.scatter(
                plot_df,
                x="sent_value",
                y="like_rate",
                color=color,
                hover_data=hover,
                labels={
                    "sent_value": "Sentimento do vídeo",
                    "like_rate": "Like rate (%)",
                },
                title="Sentimento do vídeo × Like rate",
            )

            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    # 3 — NLP

    st.header("NLP — Sentimento & Marcas")

    col1, col2 = st.columns(2)

    with col1:
        if "brand_primary" in df_f.columns:
            bc = df_f["brand_primary"].value_counts(dropna=False).reset_index()
            bc.columns = ["brand_primary", "count"]

            fig = px.bar(
                bc,
                x="brand_primary",
                y="count",
                title="Distribuição de marcas",
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "topc_label" in df_f.columns and has_comments:
            tc = (
                df_f["topc_label"]
                .fillna("unknown")
                .value_counts(dropna=False)
                .reset_index()
            )

            tc.columns = ["topc_label", "count"]

            fig = px.pie(
                tc,
                names="topc_label",
                values="count",
                title="Sentimento do comentário principal",
            )

            st.plotly_chart(fig, use_container_width=True)
        elif not has_comments:
            st.info("Execute steps 7 e 8 para ver sentimento dos comentários.")

    col3, col4 = st.columns(2)

    with col3:
        if {"brand_primary", "sent_value"}.issubset(df_f.columns):
            bs = (
                df_f.groupby("brand_primary", dropna=False)["sent_value"]
                .mean()
                .reset_index()
                .sort_values("sent_value", ascending=False)
            )

            fig = px.bar(
                bs,
                x="brand_primary",
                y="sent_value",
                title="Média de sentimento por marca",
            )

            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if {"brand_primary", "topc_value"}.issubset(df_f.columns) and has_comments:
            bc2 = (
                df_f.groupby("brand_primary", dropna=False)["topc_value"]
                .mean()
                .reset_index()
                .sort_values("topc_value", ascending=False)
            )

            fig = px.bar(
                bc2,
                x="brand_primary",
                y="topc_value",
                title="Média de sentimento do comentário por marca",
            )

            st.plotly_chart(fig, use_container_width=True)

    # 4 — TABELA COMPLETA

    st.header("Tabela analítica completa")

    cols_show = [
        c for c in [
            "videoId",
            "title",
            "channelTitle",
            "viewCount",
            "likeCount",
            "commentCount",
            "like_rate",
            "comment_rate",
            "subscriberCount",
            "brand_primary",
            "model_hint",
            "sent_label",
            "sent_value",
            "top_comment",
            "topc_label",
            "topc_value",
        ]
        if c in df_f.columns
    ]

    st.dataframe(df_f[cols_show], use_container_width=True)

    # 5 — SPAM / BOTS

    st.header("Spam & Bots — análise de comentários")

    spam_df = load_spam_data()

    if spam_df is None:
        st.info(
            "Detecção de spam ainda não executada.\n\n"
            "Execute o **Step 9** no pipeline:\n"
            "```\n"
            "python main.py\n"
            "```\n"
            "ou diretamente:\n"
            "```\n"
            "python -m pipelines.spam_detect\n"
            "```"
        )

    else:
        total_com = len(spam_df)

        if total_com == 0:
            st.warning("Coleção de comentários vazia.")
            return

        n_spam = (spam_df["spam_label"] == "spam").sum()
        n_suspeito = (spam_df["spam_label"] == "suspeito").sum()
        n_legitimo = (spam_df["spam_label"] == "legítimo").sum()

        sm1, sm2, sm3, sm4 = st.columns(4)

        with sm1:
            st.metric("Total comentários", total_com)

        with sm2:
            st.metric(
                "Spam / bots",
                f"{n_spam}  ({n_spam / total_com * 100:.1f}%)",
                delta_color="off",
            )

        with sm3:
            st.metric(
                "Suspeitos",
                f"{n_suspeito}  ({n_suspeito / total_com * 100:.1f}%)",
            )

        with sm4:
            st.metric(
                "Legítimos",
                f"{n_legitimo}  ({n_legitimo / total_com * 100:.1f}%)",
            )

        st.divider()

        sc1, sc2 = st.columns(2)

        color_map = {
            "spam": "#E24B4A",
            "suspeito": "#EF9F27",
            "legítimo": "#1D9E75",
        }

        with sc1:
            label_counts = spam_df["spam_label"].value_counts().reset_index()
            label_counts.columns = ["label", "count"]

            fig_pie = px.pie(
                label_counts,
                names="label",
                values="count",
                color="label",
                color_discrete_map=color_map,
                title="Distribuição de labels de spam",
            )

            st.plotly_chart(fig_pie, use_container_width=True)

        with sc2:
            fig_hist = px.histogram(
                spam_df,
                x="spam_score",
                color="spam_label",
                color_discrete_map=color_map,
                nbins=12,
                title="Distribuição de spam score",
                labels={
                    "spam_score": "Score",
                    "count": "Comentários",
                },
            )

            fig_hist.add_vline(
                x=3,
                line_dash="dash",
                line_color="#EF9F27",
                annotation_text="suspeito ≥ 3",
            )

            fig_hist.add_vline(
                x=5,
                line_dash="dash",
                line_color="#E24B4A",
                annotation_text="spam ≥ 5",
            )

            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("Sinais mais detectados")

        all_signals: list[str] = []
        ignored_signals = {"", "unknown", "nan", "none", "null"}

        if "spam_signals" in spam_df.columns:
            for sig_str in spam_df["spam_signals"].dropna():
                signals = [
                    s.strip()
                    for s in str(sig_str).split("|")
                    if s.strip().lower() not in ignored_signals
                ]
                all_signals.extend(signals)

        if all_signals:
            sig_counts = _Counter(all_signals).most_common(10)
            sig_df = pd.DataFrame(sig_counts, columns=["sinal", "ocorrências"])

            fig_sig = px.bar(
                sig_df,
                x="ocorrências",
                y="sinal",
                orientation="h",
                title="Top 10 sinais de spam",
                color="ocorrências",
                color_continuous_scale="Reds",
            )

            fig_sig.update_layout(yaxis={"categoryorder": "total ascending"})

            st.plotly_chart(fig_sig, use_container_width=True)
        else:
            st.info("Nenhum sinal específico de spam foi detectado.")

        if "videoId" in spam_df.columns:
            st.subheader("Spam por vídeo")

            count_col = "commentId" if "commentId" in spam_df.columns else "text"

            spam_by_video = (
                spam_df.groupby("videoId")
                .agg(
                    total=(count_col, "count"),
                    spam=("spam_label", lambda x: (x == "spam").sum()),
                    suspeito=("spam_label", lambda x: (x == "suspeito").sum()),
                )
                .reset_index()
            )

            spam_by_video["spam_rate"] = (
                (spam_by_video["spam"] + spam_by_video["suspeito"])
                / spam_by_video["total"] * 100
            ).round(1)

            spam_by_video = spam_by_video.sort_values("spam_rate", ascending=False)

            st.dataframe(
                spam_by_video.head(20),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Comentários marcados como spam")

        spam_only = spam_df[spam_df["spam_label"] == "spam"].sort_values(
            "spam_score",
            ascending=False,
        )

        cols_spam = [
            c for c in [
                "videoId",
                "author",
                "text",
                "likeCount",
                "spam_score",
                "spam_signals",
            ]
            if c in spam_only.columns
        ]

        st.dataframe(
            spam_only[cols_spam].head(50),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Ver comentários suspeitos"):
            suspeitos = spam_df[spam_df["spam_label"] == "suspeito"].sort_values(
                "spam_score",
                ascending=False,
            )

            st.dataframe(
                suspeitos[cols_spam].head(100),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()