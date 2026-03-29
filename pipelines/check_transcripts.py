# file: check_transcripts.py
import pandas as pd

df = pd.read_csv("outputs/transcripts.csv")
print("linhas:", len(df))
print(df.head(3))

# amostra curta por vídeo
for _, r in df.head(3).iterrows():
    txt = (r["text"] or "")[:200].replace("\n"," ")
    print(f"\n{r['videoId']}: {txt}...")
