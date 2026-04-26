"""
mongo_writer.py — Insere / atualiza CSVs no MongoDB usando bulk_write.

Melhoria: substitui loop de update_one (N requests) por bulk_write
(1 request), reduzindo latência 10-50x em coleções grandes.
"""

from __future__ import annotations

import pandas as pd
from pymongo import UpdateOne

from db import get_db


def insert_csv(file: str, collection: str, key_field: str = "videoId") -> None:
    print(f"\n Inserção MongoDB: {file} → {collection}")

    db = get_db()

    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print(f" [warn] Arquivo não encontrado: {file} — pulando.")
        return
    except Exception as e:
        print(f" [warn] Erro ao ler {file}: {e} — pulando.")
        return

    print(f" Linhas: {len(df)}")

    if df.empty:
        print(f" {file} vazio — nada a inserir.")
        return

    if key_field not in df.columns:
        print(f" [warn] Campo-chave '{key_field}' não encontrado em {file} — pulando.")
        return

    data = df.to_dict("records")

    ops = [
        UpdateOne(
            {key_field: item[key_field]},
            {"$set": item},
            upsert=True,
        )
        for item in data
        if key_field in item and pd.notna(item[key_field])
    ]

    if not ops:
        print(" Nenhuma operação válida gerada.")
        return

    result = db[collection].bulk_write(ops, ordered=False)

    inserted = result.upserted_count
    updated  = result.modified_count
    total    = db[collection].count_documents({})

    print(f" Inseridos: {inserted}")
    print(f" Atualizados: {updated}")
    print(f" Total MongoDB ({collection}): {total}")