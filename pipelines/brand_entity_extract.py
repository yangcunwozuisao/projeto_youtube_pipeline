import re
import pandas as pd
from pathlib import Path


def main():

    SRC = "outputs/dataset_nlp.csv"

    if not Path(SRC).exists():
        raise FileNotFoundError("dataset_nlp.csv não encontrado. Rode NLP primeiro.")

    df = pd.read_csv(SRC)

    BRANDS = {
        "samsung": [
            r"\bsamsung\b",
            r"\bgalaxy\s?(?:a|s|note|z)\b",
            r"\bgalaxy\s?a\d{1,2}\b",
            r"\bgalaxy\s?s\d{1,2}\b",
            r"\bgalaxy\s?note\b",
            r"\bgalaxy\s?note\s?\d{1,2}\b",
            r"\bz\s?fold\b",
            r"\bz\s?flip\b"
        ],
        "apple": [
            r"\bapple\b",
            r"\biphone\b",
            r"\bipad\b",
            r"\bios\b"
        ],
        "xiaomi": [
            r"\bxiaomi\b",
            r"\bredmi\b",
            r"\bpoco\b",
            r"\bmi\s?\d{1,2}\b",
            r"\bmi\s?mix\b",
            r"\bmi\s?note\b",
            r"\bmi\s?max\b"
        ],
        "motorola": [
            r"\bmotorola\b",
            r"\bmoto\s?g\b",
            r"\bmoto\s?g\d{1,2}\b",
            r"\bmoto\s?e\d{1,2}\b",
            r"\bmoto\s?edge\b",
            r"\bmotorola\s?edge\b",
            r"\bmoto\s?razr\b"
        ],
        "huawei": [
            r"\bhuawei\b",
            r"\bhuawei\s?mate\b",
            r"\bhuawei\s?p\d{2,3}\b",
            r"\bmate\s?\d{1,2}\b"
        ],
        "oneplus": [
            r"\boneplus\b"
        ],
        "oppo": [
            r"\boppo\b",
            r"\boppo\s?reno\b",
            r"\bfind\s?x\b"
        ],
        "vivo": [
            r"\bvivo\b",
            r"\bvivo\s?v\d{1,3}\b",
            r"\bvivo\s?x\d{1,3}\b",
            r"\bvivo\s?y\d{1,3}\b"
        ],
        "realme": [
            r"\brealme\b"
        ],
        "asus": [
            r"\basus\b",
            r"\brog\s?phone\b",
            r"\bzenfone\b"
        ],
        "google": [
            r"\bgoogle\b",
            r"\bgoogle\s?pixel\b"
        ],
        "infinix": [
            r"\binfinix\b",
            r"\binfinix\s?note\b",
            r"\bnote\s?40\b"
        ],
        "tecno": [
            r"\btecno\b",
            r"\btecno\s?phantom\b",
            r"\btecno\s?spark\b"
        ],
        "nothing": [
            r"\bnothing\b",
            r"\bnothing\s?phone\b"
        ]
    }

    def detect_brands(text: str):
        if not isinstance(text, str):
            return []

        t = text.lower()
        found = []

        for brand, patterns in BRANDS.items():
            for p in patterns:
                if re.search(p, t):
                    if brand not in found:
                        found.append(brand)
                    break

        return found

    MODEL_PAT = re.compile(
        r"\biphone\s?\d{1,2}(?:\s?(?:pro|max|plus))?\b|"
        r"\bgalaxy\s?(?:a|s)\d{1,2}\b|"
        r"\bgalaxy\s?note\s?\d{1,2}\b|"
        r"\bpixel\s?\d{1,2}(?:\s?(?:pro|xl|a))?\b|"
        r"\bmoto\s?g\d{1,2}\b|"
        r"\bmoto\s?edge\s?\d{1,2}\b|"
        r"\brog\s?phone\b|"
        r"\bredmagic\s?\d{0,2}\b",
        re.IGNORECASE,
    )

    def extract_model(text: str):
        if not isinstance(text, str):
            return None

        m = MODEL_PAT.search(text)
        return m.group(0) if m else None

    source_text = (
        df.get("title", pd.Series([""] * len(df))).fillna("") + " " +
        df.get("description", pd.Series([""] * len(df))).fillna("") + " " +
        df.get("text_short", pd.Series([""] * len(df))).fillna("")
    ).astype(str)

    print("[info] detectando marcas...")

    df["brands"] = source_text.apply(detect_brands)
    df["brand_count"] = df["brands"].apply(len)

    df["brand_primary"] = df["brands"].apply(
        lambda x: x[0] if len(x) > 0 else "unknown"
    )

    df["brands_text"] = df["brands"].apply(
        lambda x: ", ".join(x) if len(x) > 0 else "unknown"
    )

    df["model_hint"] = source_text.apply(extract_model)

    df_out = df[[
        "videoId",
        "title",
        "channelTitle",
        "viewCount",
        "brands",
        "brands_text",
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

    print("\nDistribuição de marcas (brand_primary):")
    print(df["brand_primary"].value_counts().head(20))

    print("\nExemplos de brands detectadas:")
    print(df[["title", "brands_text"]].head(10))


if __name__ == "__main__":
    main()