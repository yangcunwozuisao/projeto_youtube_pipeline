import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from shared_models import embedding_model, sentiment_pipeline
from keybert import KeyBERT


def main():

    SRC_VIDEOS = "outputs/videos_top50.csv"
    SRC_TRANS  = "outputs/transcripts.csv"

    if not Path(SRC_TRANS).exists():
        raise FileNotFoundError("transcripts.csv não encontrado. Rode o ASR primeiro.")

    videos = pd.read_csv(SRC_VIDEOS)
    trans  = pd.read_csv(SRC_TRANS)

    videos = videos[videos["videoId"].isin(trans["videoId"])]

    df = videos.merge(
        trans[["videoId", "language", "text"]],
        on="videoId",
        how="inner"
    )

    # DEBUG
    print("DEBUG → videos after filter:", len(videos))
    print("DEBUG → transcripts:", len(trans))
    print("DEBUG → merged:", len(df))

    if df.empty:
        print("⚠️ Nenhum dado para NLP. Verifique ASR.")
        return

    # PREPROCESS
    df["viewCount"] = pd.to_numeric(df.get("viewCount"), errors="coerce")
    df["text"] = df["text"].fillna("")

    MAX_CHARS = 3000
    df["text_short"] = df["text"].astype(str).str.slice(0, MAX_CHARS)

    print(f"[info] linhas com transcrição: {len(df)}")
    print("[info] idiomas:", df["language"].value_counts().to_dict())

    # KEYWORDS 
    kw_model = KeyBERT(model=embedding_model)

    def extract_keywords(txt: str):
        if not isinstance(txt, str) or not txt.strip():
            return ""
        try:
            kws = kw_model.extract_keywords(
                txt,
                keyphrase_ngram_range=(1, 2),
                top_n=5
            )
            return "; ".join([k for k, _ in kws])
        except Exception as e:
            print("[warn keyword]", e)
            return ""

    print("[info] extraindo keywords...")
    tqdm.pandas()
    df["keywords"] = df["text_short"].progress_apply(extract_keywords)

    # SENTIMENT
    clf = sentiment_pipeline

    label_map = {
        "LABEL_0": "NEG",
        "LABEL_1": "NEU",
        "LABEL_2": "POS"
    }

    val_map = {
        "NEG": -1,
        "NEU": 0,
        "POS": 1
    }

    def sent_score(txt: str):
        if not isinstance(txt, str) or not txt.strip():
            return {"label": "NEU", "score": 0.0, "sent_value": 0.0}

        try:
            out = clf(txt[:512])[0]
            label = label_map.get(out["label"], "NEU")
            score = float(out["score"])

            return {
                "label": label,
                "score": score,
                "sent_value": val_map[label] * score
            }
        except Exception as e:
            print("[warn sentiment]", e)
            return {"label": "NEU", "score": 0.0, "sent_value": 0.0}

    print("[info] calculando sentimento...")
    sents = df["text_short"].progress_apply(sent_score)

    df["sent_label"] = sents.apply(lambda d: d["label"])
    df["sent_conf"]  = sents.apply(lambda d: d["score"])
    df["sent_value"] = sents.apply(lambda d: d["sent_value"])

    # OUTPUT
    df_out = df.sort_values("viewCount", ascending=False)

    df_out.to_csv(
        "outputs/dataset_nlp.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\n dataset_nlp.csv salvo com {len(df_out)} linhas")


if __name__ == "__main__":
    main()