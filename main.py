from tools.mongo_writer import insert_csv
from pathlib import Path
import pandas as pd

from state_manager import is_done, mark_done
from db import get_db

print("\n Iniciação de Pipeline \n")

def check_file(path):
    if Path(path).exists():
        df = pd.read_csv(path)
        print(f" {path}: {len(df)} linhas")
    else:
        print(f" {path} NÃO EXISTE")

# 1 — Coletar videos
if not is_done("collect"):
    print(" Step 1: Collect videos")
    from tools.run_collect import run_collect
    run_collect()
    check_file("outputs/videos.csv")
    mark_done("collect")
else:
    print(" Step 1 SKIPPED")

# 2 — Filtrar
if not is_done("filter"):
    print("\n Step 2: Filter dataset")
    from tools.run_filter import run_filter
    run_filter()
    check_file("outputs/videos_clean.csv")
    mark_done("filter")
else:
    print(" Step 2 SKIPPED")

# 3 — ASR
if not is_done("asr"):
    print("\n Step 3: ASR transcription")
    from tools.run_asr import run_asr
    run_asr()
    mark_done("asr")
else:
    print(" Step 3 SKIPPED")

# 4 — NLP
if not is_done("nlp"):
    print("\n Step 4: NLP analysis")
    from tools.run_nlp import run_nlp
    run_nlp()
    check_file("outputs/dataset_nlp.csv")
    mark_done("nlp")
else:
    print(" Step 4 SKIPPED")

# 5 — Topicos
if not is_done("topics"):
    print("\n Step 5: Topic modeling")
    from tools.run_topics import run_topics
    run_topics()
    check_file("outputs/dataset_topics.csv")
    mark_done("topics")
else:
    print(" Step 5 SKIPPED")

# 6 — Marcas
if not is_done("brand"):
    print("\n Step 6: Brand extraction")
    from tools.run_brand import run_brand
    run_brand()
    check_file("outputs/dataset_brands.csv")
    mark_done("brand")
else:
    print(" Step 6 SKIPPED")

# 7 — Comentarios
if not is_done("comments"):
    print("\n Step 7: Comment analysis")
    from tools.run_comments import run_comments
    run_comments()
    check_file("outputs/comments.csv")
    mark_done("comments")
else:
    print(" Step 7 SKIPPED")

# 8 — Analise
if not is_done("analytics"):
    print("\n Step 8: Analytics export")
    from tools.run_analytics import run_analytics
    run_analytics()
    mark_done("analytics")
else:
    print(" Step 8 SKIPPED")

print("\n Pipeline finished successfully!\n")

# MongoDB
if not is_done("mongo"):
    print("\n MongoDB:\n")
    insert_csv("outputs/videos_clean.csv", "videos", key_field="videoId")
    insert_csv("outputs/transcripts.csv", "transcripts", key_field="videoId")
    insert_csv("outputs/dataset_nlp.csv", "nlp", key_field="videoId")
    insert_csv("outputs/dataset_topics.csv", "topics", key_field="videoId")
    insert_csv("outputs/dataset_brands.csv", "brands", key_field="videoId")

    # comments usa commentId
    insert_csv("outputs/comments.csv", "comments", key_field="commentId")

    mark_done("mongo")
else:
    print(" MongoDB SKIPPED")

# MongoDB check
db = get_db()

print("\n MongoDB collections:\n")

cols = db.list_collection_names()

if not cols:
    print(" MongoDB :Nenhuma coleção encontrada")
else:
    for col in cols:
        count = db[col].count_documents({})
        print(f"{col} → {count}")