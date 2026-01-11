# app/db.py
import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
DOC_COLLECTION = os.getenv("MONGODB_COLLECTION", "marketing_texts")
SENT_COLLECTION = os.getenv("MONGODB_SENTENCE_COLLECTION", "marketing_sentence")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")
if not MONGODB_DB:
    raise RuntimeError("MONGODB_DB is not set")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

documents = db[DOC_COLLECTION]
sentence_analysis = db[SENT_COLLECTION]

print("MONGO URI: ", MONGODB_URI)
print("MONGO DB:", MONGODB_DB)
print("DOC COL :", documents.name)
print("SENT COL:", sentence_analysis.name)