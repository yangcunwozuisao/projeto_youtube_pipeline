import re
import pandas as pd
from pathlib import Path
from shared_models import get_ner_model

LABELS = ["brand", "series", "model"]

BRAND_CANON = {
    "samsung": ["samsung", "galaxy"],
    "xiaomi": ["xiaomi", "redmi", "poco"],
    "apple": ["apple", "iphone", "ipad"],
    "motorola": ["motorola", "moto"],
    "huawei": ["huawei"],
    "realme": ["realme"],
    "oneplus": ["oneplus", "one plus"],
}

SERIES_PATTERNS = [
    r"\bgalaxy\s?[aszmf]\d{1,3}\b",
    r"\bgalaxy\s?s\d{1,2}\s?(?:ultra|plus|fe)?\b",
    r"\bredmi\s?note\s?\d{1,2}\s?(?:pro|max|plus)?\b",
    r"\biphone\s?\d{1,2}\s?(?:pro|max|plus|mini)?\b",
    r"\bmoto\s?g\d+\b",
]

MODEL_PATTERNS = [
    r"\bs\d{1,2}\s?(?:ultra|plus|fe)?\b",
    r"\ba\d{1,2}\b",
    r"\bnote\s?\d{1,2}\s?(?:pro|max|plus)?\b",
    r"\biphone\s?\d{1,2}\s?(?:pro|max|plus|mini)?\b",
    r"\bg\d{1,3}\b",
]

def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()

def build_text(row):
    parts = [
        norm_text(row.get("title")),
        norm_text(row.get("text_short")),
        norm_text(row.get("text")),
        norm_text(row.get("top_comment")),
        norm_text(row.get("channelTitle")),
    ]
    text = " | ".join([p for p in parts if p])
    return text[:4000]

def rule_brand(text):
    t = text.lower()
    hits = []
    for brand, words in BRAND_CANON.items():
        for w in words:
            if w in t:
                hits.append(brand)
                break
    if hits:
        return hits[0], hits
    return None, []

def rule_series(text):
    t = text.lower()
    found = []
    for p in SERIES_PATTERNS:
        found += re.findall(p, t, flags=re.I)
    return list(dict.fromkeys(found))

def rule_model(text):
    t = text.lower()
    found = []
    for p in MODEL_PATTERNS:
        found += re.findall(p, t, flags=re.I)
    return list(dict.fromkeys(found))

def ner_extract(text):
    if not text.strip():
        return []

    ner_model = get_ner_model()

    ents = ner_model.predict_entities(
        text,
        LABELS,
        threshold=0.45
    )
    return ents

def normalize_brand(x):
    if not x:
        return None
    x = x.lower().strip()
    for brand, aliases in BRAND_CANON.items():
        if x == brand or x in aliases:
            return brand
    return x

def extract_brand_fields(text):
    # 1) Prioridade regra
    rule_primary, rule_hits = rule_brand(text)
    series_hits = rule_series(text)
    model_hits = rule_model(text)

    # 2) NER adicao
    ents = ner_extract(text)

    ner_brands = []
    ner_series = []
    ner_models = []
    ner_scores = []

    for e in ents:
        label = str(e.get("label", "")).lower()
        etext = str(e.get("text", "")).strip()
        score = float(e.get("score", 0))

        if not etext:
            continue

        if label == "brand":
            ner_brands.append(normalize_brand(etext))
            ner_scores.append(score)
        elif label == "series":
            ner_series.append(etext)
            ner_scores.append(score)
        elif label == "model":
            ner_models.append(etext)
            ner_scores.append(score)

    ner_brands = [x for x in ner_brands if x]
    ner_brands = list(dict.fromkeys(ner_brands))
    ner_series = list(dict.fromkeys(ner_series))
    ner_models = list(dict.fromkeys(ner_models))

    # 3) juntar
    brand_primary = (
        rule_primary
        or (ner_brands[0] if ner_brands else None)
        or "unknown"
    )

    brand_series = (
        (series_hits[0] if series_hits else None)
        or (ner_series[0] if ner_series else None)
        or "unknown"
    )

    model_hint = (
        (model_hits[0] if model_hits else None)
        or (ner_models[0] if ner_models else None)
        or "unknown"
    )

    all_brands = list(dict.fromkeys(rule_hits + ner_brands))
    brands_text = ", ".join(all_brands) if all_brands else "unknown"

    if rule_primary:
        brand_source = "rule"
    elif ner_brands or ner_series or ner_models:
        brand_source = "ner"
    else:
        brand_source = "fallback"

    brand_conf = max(ner_scores) if ner_scores else 0.0

    return {
        "brand_primary": brand_primary,
        "brand_series": brand_series,
        "model_hint": model_hint,
        "brands_text": brands_text,
        "brand_source": brand_source,
        "brand_conf": brand_conf,
    }

def main():
    src_candidates = [
    "outputs/dataset_enriched.csv",
    "outputs/dataset_topics.csv",
    "outputs/dataset_nlp.csv",
    "outputs/videos_clean.csv",
    ]

    src = None
    for p in src_candidates:
        if Path(p).exists():
            src = p
            break

    if src is None:
        raise FileNotFoundError("Nenhuma base encontrada para extracao de marcas.")

    df = pd.read_csv(src)

    rows = []
    for _, row in df.iterrows():
        text = build_text(row)
        info = extract_brand_fields(text)
        rows.append(info)

    out = pd.DataFrame(rows)
    df_final = pd.concat([df.reset_index(drop=True), out], axis=1)

    Path("outputs").mkdir(exist_ok=True)
    df_final.to_csv("outputs/dataset_brands.csv", index=False)

    print("dataset_brands.csv gerado com sucesso")