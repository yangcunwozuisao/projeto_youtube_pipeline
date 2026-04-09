from db import get_db
import pandas as pd


def insert_csv(file, collection, key_field="videoId"):

    print(f"\n Insercao MongoDB: {file} → {collection}")

    db = get_db()

    df = pd.read_csv(file)

    print(" Linha:", len(df))

    if df.empty:
        print(f" {file} vazio")
        return

    data = df.to_dict("records")

    inserted = 0
    updated = 0

    for item in data:

        if key_field not in item:
            continue

        result = db[collection].update_one(
            {key_field: item[key_field]}, 
            {"$set": item},
            upsert=True
        )

        if result.upserted_id is not None:
            inserted += 1
        else:
            updated += 1

    total = db[collection].count_documents({})

    print(f" Inseridos: {inserted}")
    print(f" Atualizados: {updated}")
    print(f" Total MongoDB ({collection}): {total}")