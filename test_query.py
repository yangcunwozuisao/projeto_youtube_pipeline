from pymongo import MongoClient
import pandas as pd

# conectar MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_db"]

data = list(db.videos.find())

df = pd.DataFrame(data)

print("Número da entrada de dados:", len(df))
print(df.head())