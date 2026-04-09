import streamlit as st
import pandas as pd
from db import get_db

st.set_page_config(page_title="YouTube Analysis", layout="wide")

st.title("YouTube Brand Analysis Dashboard")

db = get_db()
data = list(db["brands"].find())
df = pd.DataFrame(data)

if df.empty:
    st.warning("Sem dado")
    st.stop()

st.write(f"Total registros: {len(df)}")

# 调试：先看字段
st.subheader("DEBUG - colunas")
st.write(df.columns.tolist())

# 调试：看原始情感字段
debug_cols = [c for c in ["brand_primary", "sent_label", "sent_value", "title"] if c in df.columns]
st.subheader("DEBUG - amostra bruta")
st.dataframe(df[debug_cols].head(20))

# 强制转数值
if "sent_value" in df.columns:
    df["sent_value"] = pd.to_numeric(df["sent_value"], errors="coerce")

if "viewCount" in df.columns:
    df["viewCount"] = pd.to_numeric(df["viewCount"], errors="coerce")

# 调试：看转换后分布
st.subheader("DEBUG - sent_value describe")
if "sent_value" in df.columns:
    st.write(df["sent_value"].describe())

st.subheader("DEBUG - sent_label counts")
if "sent_label" in df.columns:
    st.write(df["sent_label"].value_counts(dropna=False))

st.subheader("DEBUG - brand_primary counts")
if "brand_primary" in df.columns:
    st.write(df["brand_primary"].value_counts(dropna=False))

st.subheader("Marcas")
if "brand_primary" in df.columns:
    brand_counts = df["brand_primary"].fillna("null").value_counts()
    st.bar_chart(brand_counts)

st.subheader("Sentimentos")
if "brand_primary" in df.columns and "sent_value" in df.columns:
    sentiment = (
        df.assign(brand_primary=df["brand_primary"].fillna("null"))
          .groupby("brand_primary", dropna=False)["sent_value"]
          .mean()
          .dropna()
          .reset_index()
    )

    st.subheader("DEBUG - tabela de sentimento")
    st.dataframe(sentiment)

    if not sentiment.empty:
        st.bar_chart(sentiment.set_index("brand_primary"))
    else:
        st.warning("Tabela de sentimento vazia após conversão numérica.")

st.subheader("Videos")
if "viewCount" in df.columns:
    top_videos = df.sort_values("viewCount", ascending=False).head(10)
    st.dataframe(top_videos)

st.subheader("Filter by Brand")
if "brand_primary" in df.columns:
    options = df["brand_primary"].fillna("null").unique().tolist()
    brand = st.selectbox("Escolha a marca", options)
    filtered = df[df["brand_primary"].fillna("null") == brand]
    st.write(filtered)

st.subheader("Tendencia")
if "publishedAt" in df.columns and "sent_value" in df.columns and "brand_primary" in df.columns:
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")

    trend = (
        df.assign(brand_primary=df["brand_primary"].fillna("null"))
          .dropna(subset=["publishedAt"])
          .groupby([
              pd.Grouper(key="publishedAt", freq="W"),
              "brand_primary"
          ])["sent_value"]
          .mean()
          .reset_index()
    )

    st.subheader("DEBUG - trend")
    st.dataframe(trend.head(20))

    if not trend.empty:
        pivot = trend.pivot(
            index="publishedAt",
            columns="brand_primary",
            values="sent_value"
        )
        st.line_chart(pivot)