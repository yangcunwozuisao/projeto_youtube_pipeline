import pandas as pd

def main():

    df = pd.read_csv("outputs/videos_clean.csv")

    df = df[df["duration_sec"] >= 30] 

    df = df.sort_values("viewCount", ascending=False)

    df_top50 = df.head(50)

    print(f"Base filtrada: {len(df)} linhas | Top50: {len(df_top50)}")

    df.to_csv("outputs/videos_filtered.csv", index=False, encoding="utf-8-sig")
    df_top50.to_csv("outputs/videos_top50.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    main()
