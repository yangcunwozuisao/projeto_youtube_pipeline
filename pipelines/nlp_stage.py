"""
nlp_stage.py — Extração de keywords e sentimento das transcrições.

Correção: o código anterior tinha SRC_VIDEOS = "outputs/videos_top50.csv"
hard-coded sem fallback, causando crash se o arquivo não existisse.
Agora busca a primeira fonte disponível numa lista de prioridade.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from shared_models import get_embedding_model, get_sentiment_pipeline
from keybert import KeyBERT


# ── Constantes ────────────────────────────────────────────────────────────────

VIDEO_SOURCES = [
    "outputs/videos_top50.csv",
    "outputs/videos_clean.csv",
    "outputs/videos_filtered.csv",
    "outputs/videos.csv",
]

SRC_TRANS = "outputs/transcripts.csv"
MAX_CHARS = 3000

LABEL_MAP = {
    "LABEL_0": "NEG", "LABEL_1": "NEU", "LABEL_2": "POS",
    "Negative": "NEG", "Neutral": "NEU", "Positive": "POS",
    "negative": "NEG", "neutral": "NEU", "positive": "POS",
    "NEG":      "NEG", "NEU":     "NEU", "POS":      "POS",
}
VAL_MAP = {"NEG": -1, "NEU": 0, "POS": 1}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:

    # Fonte de vídeos: primeira disponível na lista de prioridade
    src_videos = next((p for p in VIDEO_SOURCES if Path(p).exists()), None)
    if src_videos is None:
        raise FileNotFoundError(
            "Nenhum arquivo de vídeos encontrado para NLP.\n"
            "Esperados (em ordem): " + ", ".join(VIDEO_SOURCES)
        )
    print(f"[info] fonte de vídeos: {src_videos}")

    if not Path(SRC_TRANS).exists():
        raise FileNotFoundError(
            "transcripts.csv não encontrado. Rode o step de ASR primeiro."
        )

    videos = pd.read_csv(src_videos)
    trans  = pd.read_csv(SRC_TRANS)

    videos = videos[videos["videoId"].isin(trans["videoId"])]

    df = videos.merge(
        trans[["videoId", "language", "text"]],
        on="videoId",
        how="inner",
    )

    print(f"[debug] vídeos após filtro: {len(videos)}")
    print(f"[debug] transcrições:       {len(trans)}")
    print(f"[debug] merged:             {len(df)}")

    if df.empty:
        print("[warn] Nenhum dado para NLP. Verifique o step de ASR.")
        return

    df["viewCount"] = pd.to_numeric(df.get("viewCount"), errors="coerce")
    df["text"]      = df["text"].fillna("")
    df["text_short"] = df["text"].astype(str).str.slice(0, MAX_CHARS)

    print(f"[info] linhas com transcrição: {len(df)}")
    print("[info] idiomas:", df["language"].value_counts().to_dict())

    # ── Keywords ──────────────────────────────────────────────────────────────
    kw_model = KeyBERT(model=get_embedding_model())

    def extract_keywords(txt: str) -> str:
        if not isinstance(txt, str) or not txt.strip():
            return ""
        try:
            kws = kw_model.extract_keywords(txt, keyphrase_ngram_range=(1, 2), top_n=5)
            return "; ".join([k for k, _ in kws])
        except Exception as e:
            print("[warn keyword]", e)
            return ""

    print("[info] extraindo keywords...")
    tqdm.pandas()
    df["keywords"] = df["text_short"].progress_apply(extract_keywords)

    # ── Sentimento ────────────────────────────────────────────────────────────
    clf = get_sentiment_pipeline()
    _label_logged = False

    def sent_score(txt: str) -> dict:
        nonlocal _label_logged
        if not isinstance(txt, str) or not txt.strip():
            return {"label": "NEU", "score": 0.0, "sent_value": 0.0}
        try:
            out = clf(txt[:512])[0]
            if not _label_logged:
                print(f"[debug] label bruto do modelo: {out['label']!r}  score: {out['score']:.3f}")
                _label_logged = True
            label = LABEL_MAP.get(out["label"], "NEU")
            score = float(out["score"])
            return {"label": label, "score": score, "sent_value": VAL_MAP[label] * score}
        except Exception as e:
            print("[warn sentiment]", e)
            return {"label": "NEU", "score": 0.0, "sent_value": 0.0}

    print("[info] calculando sentimento...")
    sents = df["text_short"].progress_apply(sent_score)

    df["sent_label"] = sents.apply(lambda d: d["label"])
    df["sent_conf"]  = sents.apply(lambda d: d["score"])
    df["sent_value"] = sents.apply(lambda d: d["sent_value"])

    # ── Output ────────────────────────────────────────────────────────────────
    df_out = df.sort_values("viewCount", ascending=False)
    df_out.to_csv("outputs/dataset_nlp.csv", index=False, encoding="utf-8-sig")

    print(f"\n dataset_nlp.csv salvo com {len(df_out)} linhas")


if __name__ == "__main__":
    main()