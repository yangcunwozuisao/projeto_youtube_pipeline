"""
Detecção de comentários suspeitos / bots

Estratégia híbrida:
  1. Sinais de regra  (rápido, sem modelo)
  2. Similaridade de embeddings  (usa SentenceTransformer já carregado)

Saída: colunas adicionadas ao DataFrame de comentários
  spam_score   — int  (0 = legítimo … alto = muito suspeito)
  spam_label   — str  ("legítimo" | "suspeito" | "spam")
  spam_signals — str  (lista dos sinais disparados, para auditoria)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


# Entrada

SCORE_SUSPEITO = 3     # score >= 3 → suspeito
SCORE_SPAM     = 5     # score >= 5 → spam/bot

SIM_THRESHOLD  = 0.92  # similaridade coseno mínima para "quase-duplicata"
MIN_NEAR_DUP_GROUP = 2  # grupo precisa ter >= N comentários para disparar

# Palavras-chave promocionais

PROMO_PATTERNS = re.compile(
    r"""
    check[\s_]*(out[\s_]*)?my[\s_]*(channel|page|profile|link)|
    sub[\s_]*4[\s_]*sub|s4s|
    follow[\s_]*me|
    subscribe[\s_]*(to[\s_]*)?my|
    giveaway|
    free[\s_]*(gift|iphone|samsung|prize)|
    i[\s]*give[\s]*away|
    link[\s]*in[\s]*bio|
    click[\s]*(the[\s]*)?link|
    visit[\s]*my|
    promo[\s]*code|
    use[\s]*code|
    discount|desconto|
    ganhe[\s]*grátis|
    clique[\s]*no[\s]*link
    """,
    re.IGNORECASE | re.VERBOSE,
)

URL_PATTERN = re.compile(
    r"(https?://|www\.|t\.me/|wa\.me/|bit\.ly/|youtu\.be/)",
    re.IGNORECASE,
)

#Proporção de caracteres que são emoji
def _emoji_density(text: str) -> float:
    if not text:
        return 0.0
    emoji_count = sum(
        1 for ch in text
        if unicodedata.category(ch) in ("So", "Sm", "Sk")
        or ord(ch) > 0x1F300
    )
    return emoji_count / max(len(text), 1)


#True se tiver >= 4 caracteres idênticos consecutivos (ex: 'kkkk', '!!!!').
def _repetitive_chars(text: str) -> bool:
    return bool(re.search(r"(.)\1{3,}", text))


def _caps_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def _clean_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# Sinais por comentário

def _score_row(
    row: dict,
    author_comment_counts: Counter,
    author_zero_likes_counts: Counter,
    duplicate_texts: set[str],
) -> tuple[int, list[str]]:

    #Retorna (score, lista_de_sinais) para um único comentário.
    text      = str(row.get("text") or "")
    author    = str(row.get("author") or "")
    like_count = int(row.get("likeCount") or 0)
    score     = 0
    signals: list[str] = []

    # Sinais de regra

    # 1. URL / link externo
    if URL_PATTERN.search(text):
        score += 3
        signals.append("url")

    # 2. Palavras-chave promocionais
    if PROMO_PATTERNS.search(text):
        score += 3
        signals.append("promo_keywords")

    # 3. Texto muito curto (ignorando espaços)
    if 0 < len(text.replace(" ", "")) < 15:
        score += 1
        signals.append("very_short")

    # 4. Alta densidade de emoji
    if _emoji_density(text) > 0.30:
        score += 2
        signals.append("emoji_spam")

    # 5. Caracteres repetitivos
    if _repetitive_chars(text):
        score += 1
        signals.append("repetitive_chars")

    # 6. Alto ratio de maiúsculas (> 60 %)
    if _caps_ratio(text) > 0.60 and len(text) > 10:
        score += 1
        signals.append("all_caps")

    # 7. Duplicata exata no dataset
    clean = _clean_for_dedup(text)
    if clean in duplicate_texts:
        score += 3
        signals.append("exact_duplicate")

    # 8. Mesmo autor com muitos comentários (>= 3)
    if author_comment_counts.get(author, 0) >= 3:
        score += 2
        signals.append("same_author_many")

    # 9. Mesmo autor com muitos comentários de 0 likes
    if author_zero_likes_counts.get(author, 0) >= 3:
        score += 1
        signals.append("zero_likes_author")

    return score, signals

# Similaridade de embeddings

def _find_near_duplicates(
    texts: list[str],
    threshold: float = SIM_THRESHOLD,
    min_group: int = MIN_NEAR_DUP_GROUP,
) -> set[int]:
    """
    Retorna índices de comentários que fazem parte de um grupo
    de quase-duplicatas (similaridade coseno >= threshold).

    Usa batch matrix multiplication — eficiente para até ~50K comentários.
    """
    try:
        from shared_models import embedding_model
    except ImportError:
        print("[warn spam] shared_models não disponível — pulando near-duplicate check")
        return set()

    print(f"[spam] gerando embeddings para {len(texts)} comentários...")
    embs = embedding_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # L2-norm → dot product = cosine similarity
    ).astype(np.float32)

    suspicious_idx: set[int] = set()
    batch = 512   # processa em blocos para não explodir a memória

    for start in tqdm(range(0, len(embs), batch), desc="[spam] similaridade"):
        end   = min(start + batch, len(embs))
        chunk = embs[start:end]                      # (B, D)
        sims  = chunk @ embs.T                       # (B, N)

        for local_i, global_i in enumerate(range(start, end)):
            row_sims = sims[local_i]
            # índices similares (excluindo a si mesmo)
            similar = np.where(
                (row_sims >= threshold) & (np.arange(len(embs)) != global_i)
            )[0]
            if len(similar) >= min_group - 1:
                suspicious_idx.add(global_i)
                suspicious_idx.update(similar.tolist())

    return suspicious_idx

def main(use_embeddings: bool = True):
    """
    Lê outputs/comments.csv, adiciona colunas de spam e salva
    outputs/comments_spam.csv.

    Parâmetros:
        use_embeddings: se True, roda a detecção de quase-duplicatas
                        via embeddings (mais lento mas mais preciso).
                        Se False, usa apenas sinais de regra.
    """
    SRC = Path("outputs/comments.csv")
    OUT = Path("outputs/comments_spam.csv")

    if not SRC.exists():
        raise FileNotFoundError(
            "outputs/comments.csv não encontrado. Execute o step de comentários primeiro."
        )

    df = pd.read_csv(SRC)
    print(f"[spam] {len(df)} comentários carregados")

    df["text"]      = df["text"].fillna("").astype(str)
    df["author"]    = df["author"].fillna("").astype(str)
    df["likeCount"] = pd.to_numeric(df.get("likeCount"), errors="coerce").fillna(0).astype(int)

    author_counts = Counter(df["author"].tolist())

    # autores com >= 3 comentários e 0 likes em todos eles
    zero_likes_df = df[df["likeCount"] == 0]
    author_zero   = Counter(zero_likes_df["author"].tolist())
    # só conta se também tem >= 3 comentários no total
    author_zero_suspicious = Counter(
        {a: c for a, c in author_zero.items() if c >= 3}
    )

    # textos que aparecem 2+ vezes (duplicatas exatas)
    clean_texts = df["text"].apply(_clean_for_dedup)
    text_counts = Counter(clean_texts.tolist())
    duplicate_set = {t for t, c in text_counts.items() if c >= 2 and t.strip()}

    print(f"[spam] {len(duplicate_set)} textos duplicados encontrados")

    print("[spam] aplicando sinais de regra...")
    records   = df.to_dict("records")
    scores    = []
    signals_l = []

    for row in tqdm(records, desc="[spam] regras"):
        s, sigs = _score_row(row, author_counts, author_zero_suspicious, duplicate_set)
        scores.append(s)
        signals_l.append(sigs)

    df["spam_score"]   = scores
    df["spam_signals"] = ["|".join(s) if s else "" for s in signals_l]

    if use_embeddings:
        near_dup_idx = _find_near_duplicates(df["text"].tolist())
        print(f"[spam] {len(near_dup_idx)} comentários em grupos de quase-duplicatas")

        for idx in near_dup_idx:
            df.at[idx, "spam_score"] += 2
            existing = df.at[idx, "spam_signals"]
            df.at[idx, "spam_signals"] = (
                existing + "|near_duplicate" if existing else "near_duplicate"
            )

    def _label(score: int) -> str:
        if score >= SCORE_SPAM:
            return "spam"
        if score >= SCORE_SUSPEITO:
            return "suspeito"
        return "legítimo"

    df["spam_label"] = df["spam_score"].apply(_label)

    counts = df["spam_label"].value_counts()
    total  = len(df)
    print("\n── Resultado da detecção ──────────────────────────────")
    for label, count in counts.items():
        pct = count / total * 100
        print(f"  {label:<12} {count:>6}  ({pct:.1f}%)")
    print("────────────────────────────────────────────────────────\n")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"[spam] salvo: {OUT}  ({len(df)} linhas)")

    return df


if __name__ == "__main__":
    import sys
    use_emb = "--no-embeddings" not in sys.argv
    main(use_embeddings=use_emb)