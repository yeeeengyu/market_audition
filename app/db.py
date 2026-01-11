import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
MONGODB_COLLECTION = os.getenv("MONGODB_COLLECTION", "marketing_texts")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")

if not MONGODB_DB:
    raise RuntimeError("MONGODB_DB is not set")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]
documents = db[MONGODB_COLLECTION]

print("MONGO URI:", MONGODB_URI)
print("MONGO DB :", MONGODB_DB)
print("COLLECTION:", documents.name)
