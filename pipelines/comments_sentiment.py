# file: comments_sentiment.py

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from shared_models import sentiment_pipeline


def main():

    SRC_DATA = (
        "outputs/dataset_topics.csv"
        if Path("outputs/dataset_topics.csv").exists()
        else "outputs/dataset_nlp.csv"
    )
    SRC_COM = "outputs/comments.csv"
    OUT_FILE = "outputs/dataset_enriched.csv"

    if not Path(SRC_DATA).exists():
        raise FileNotFoundError("Base analítica não encontrada.")

    if not Path(SRC_COM).exists():
        raise FileNotFoundError("comments.csv não encontrado. Rode yt_collect_comments.py")

    df = pd.read_csv(SRC_DATA)
    cm = pd.read_csv(SRC_COM)

    cm["likeCount"] = pd.to_numeric(cm.get("likeCount"), errors="coerce").fillna(0)

    topc = (
        cm.sort_values(["videoId", "likeCount"], ascending=[True, False])
        .drop_duplicates("videoId")[["videoId", "text", "likeCount"]]
        .rename(columns={"text": "top_comment", "likeCount": "topc_likes"})
    )

    clf = sentiment_pipeline

    label_map = {
        "LABEL_0": "NEG",
        "LABEL_1": "NEU",
        "LABEL_2": "POS",
        "NEG": "NEG",
        "NEU": "NEU",
        "POS": "POS"
    }

    val_map = {
        "NEG": -1,
        "NEU": 0,
        "POS": 1
    }

    def sent_score(txt: str):
        if not isinstance(txt, str) or not txt.strip():
            return {"label": "NEU", "score": 0.0, "value": 0.0}

        out = clf(txt[:512])[0]
        lab = label_map.get(out["label"], "NEU")
        sc = float(out["score"])

        return {
            "label": lab,
            "score": sc,
            "value": val_map[lab] * sc
        }

    tqdm.pandas()

    sc = topc["top_comment"].progress_apply(sent_score)

    topc["topc_label"] = sc.apply(lambda d: d["label"])
    topc["topc_conf"] = sc.apply(lambda d: d["score"])
    topc["topc_value"] = sc.apply(lambda d: d["value"])

    out = df.merge(topc, on="videoId", how="left")

    out.to_csv(
        OUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"salvo {OUT_FILE} com {len(out)} linhas")


if __name__ == "__main__":
    main()