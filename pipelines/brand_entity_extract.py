import re
import pandas as pd
from pathlib import Path


def main():

    SRC = "outputs/dataset_nlp.csv"

    if not Path(SRC).exists():
        raise FileNotFoundError("dataset_nlp.csv não encontrado. Rode NLP primeiro.")

    df = pd.read_csv(SRC)

    # Regra p/ marcas
    BRANDS = {
        "samsung": ["samsung", r"\bgalaxy\b"],
        "apple": ["apple", r"\biphone\b", r"\bipad\b"],
        "xiaomi": ["xiaomi", r"\bredmi\b", r"\bpoco\b"],
        "motorola": ["motorola", r"\bmoto\b"],
        "huawei": ["huawei", r"\bmate\b"],
        "oneplus": ["oneplus"],
        "oppo": ["oppo"],
        "vivo": ["vivo"],
        "realme": ["realme"],
    }

    def detect_brands(text: str):

        if not isinstance(text, str):
            return []

        t = text.lower()
        found = []

        for brand, patterns in BRANDS.items():
            for p in patterns:
                if re.search(p, t):
                    found.append(brand)
                    break

        return list(set(found)) 

    MODEL_PAT = re.compile(
        r"\b(?:a|s)\d{2}\b|\b\d{1,2}\s?(?:pro|ultra|plus|max)\b|\biphone\s?\d{1,2}\b",
        re.IGNORECASE,
    )

    def extract_model(text: str):
        if not isinstance(text, str):
            return None

        m = MODEL_PAT.search(text)
        return m.group(0) if m else None

    # TEXT SOURCE
    source_text = (
        df["title"].fillna("") + " " + df["text_short"].fillna("")
    ).astype(str)

    print("[info] detectando marcas...")

    df["brands"] = source_text.apply(detect_brands)

    df["brand_count"] = df["brands"].apply(len)

    df["brand_primary"] = df["brands"].apply(
        lambda x: x[0] if len(x) > 0 else None
    )

    # MODEL
    df["model_hint"] = source_text.apply(extract_model)

    # OUTPUT
    df_out = df[[
        "videoId",
        "title",
        "channelTitle",
        "viewCount",
        "brands",
        "brand_primary",
        "brand_count",
        "model_hint",
        "sent_label",
        "sent_value"
    ]]

    df_out.to_csv(
        "outputs/dataset_brands.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(f"\ndataset_brands.csv salvo com {len(df_out)} linhas")

    print("\nDistribuição de marcas:")
    print(df["brand_primary"].value_counts().head(10))


if __name__ == "__main__":
    main()