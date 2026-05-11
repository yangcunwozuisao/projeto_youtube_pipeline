from pathlib import Path
import pandas as pd

from shared_models import sentiment_pipeline


LABEL_MAP = {
    "LABEL_0": "NEG", "LABEL_1": "NEU", "LABEL_2": "POS",
    "Negative": "NEG", "Neutral": "NEU", "Positive": "POS",
    "negative": "NEG", "neutral": "NEU", "positive": "POS",
    "NEG": "NEG", "NEU": "NEU", "POS": "POS",
}
VAL_MAP = {"NEG": -1, "NEU": 0, "POS": 1}


def sent_score(txt: str):
    if not isinstance(txt, str) or not txt.strip():
        return {"label": "NEU", "score": 0.0, "sent_value": 0.0}

    try:
        out = sentiment_pipeline(txt[:512])[0]
        label = LABEL_MAP.get(out["label"], "NEU")
        score = float(out["score"])
        return {
            "label": label,
            "score": score,
            "sent_value": VAL_MAP[label] * score,
        }
    except Exception as e:
        print("[warn sentiment comment]", e)
        return {"label": "NEU", "score": 0.0, "sent_value": 0.0}


def load_base_dataset():
    candidates = [
        "outputs/dataset_topics.csv",
        "outputs/dataset_nlp.csv",
        "outputs/videos_top50.csv",
    ]

    for p in candidates:
        if Path(p).exists():
            print(f"[info] base analítica para comentários: {p}")
            return pd.read_csv(p)

    raise FileNotFoundError(
        "Nenhuma base encontrada para enriquecer comentários. "
        "Rode NLP/topics primeiro."
    )


def pick_top_comment(df_comments: pd.DataFrame) -> pd.DataFrame:
    df = df_comments.copy()

    if "text" not in df.columns:
        raise ValueError("comments.csv não possui a coluna 'text'.")

    df["videoId"] = df["videoId"].astype(str)
    df["text"] = df["text"].fillna("").astype(str)
    df["likeCount"] = pd.to_numeric(df.get("likeCount"), errors="coerce").fillna(0)
    df["publishedAt_dt"] = pd.to_datetime(df.get("publishedAt"), errors="coerce")

    if "isReply" in df.columns:
        df["isReply"] = df["isReply"].fillna(False).astype(bool)
    else:
        df["isReply"] = False

    # somente comentario como string
    df = df[df["text"].str.strip() != ""].copy()

    if df.empty:
        return pd.DataFrame(columns=["videoId", "top_comment", "topc_likes"])

    # Prioridade:
    # 1. Comentários que não sejam respostas do outro comentario.
    # 2. likeCount maior
    # 3. mais tempo que esta no video
    df = df.sort_values(
        by=["videoId", "isReply", "likeCount", "publishedAt_dt"],
        ascending=[True, True, False, True],
        na_position="last"
    )

    top_df = df.drop_duplicates(subset=["videoId"], keep="first").copy()

    top_df = top_df.rename(columns={
        "text": "top_comment",
        "likeCount": "topc_likes"
    })

    return top_df[["videoId", "top_comment", "topc_likes"]]


def main():
    comments_path = Path("outputs/comments.csv")
    if not comments_path.exists():
        raise FileNotFoundError(
            "outputs/comments.csv não encontrado. Execute a coleta de comentários primeiro."
        )

    base_df = load_base_dataset()
    comments_df = pd.read_csv(comments_path)

    if "videoId" not in comments_df.columns:
        raise ValueError("comments.csv não possui a coluna 'videoId'.")

    if comments_df.empty:
        print("[warn] comments.csv vazio. Gerando dataset_enriched.csv sem colunas topc_*.")
        out = base_df.copy()
        out["top_comment"] = None
        out["topc_likes"] = None
        out["topc_label"] = None
        out["topc_conf"] = None
        out["topc_value"] = None
        out.to_csv("outputs/dataset_enriched.csv", index=False, encoding="utf-8-sig")
        print("[ok] dataset_enriched.csv salvo (sem comentários)")
        return

    top_comments = pick_top_comment(comments_df)

    if top_comments.empty:
        print("[warn] Nenhum comentário textual válido encontrado.")
        out = base_df.copy()
        out["top_comment"] = None
        out["topc_likes"] = None
        out["topc_label"] = None
        out["topc_conf"] = None
        out["topc_value"] = None
        out.to_csv("outputs/dataset_enriched.csv", index=False, encoding="utf-8-sig")
        print("[ok] dataset_enriched.csv salvo (sem comentários válidos)")
        return

    print(f"[info] vídeos com top comment: {len(top_comments)}")

    sent_results = top_comments["top_comment"].apply(sent_score)
    top_comments["topc_label"] = sent_results.apply(lambda d: d["label"])
    top_comments["topc_conf"] = sent_results.apply(lambda d: d["score"])
    top_comments["topc_value"] = sent_results.apply(lambda d: d["sent_value"])

    base_df["videoId"] = base_df["videoId"].astype(str)
    enriched = base_df.merge(
        top_comments.drop_duplicates("videoId"),
        on="videoId",
        how="left"
    )

    enriched.to_csv(
        "outputs/dataset_enriched.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"[ok] dataset_enriched.csv salvo com {len(enriched)} linhas")


if __name__ == "__main__":
    main()