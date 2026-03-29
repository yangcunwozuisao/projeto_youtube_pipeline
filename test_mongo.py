from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["youtube_db"]

print("✅ conectado ao MongoDB")