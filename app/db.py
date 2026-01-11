# app/db.py
import os
from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DB = os.getenv("MONGODB_DB")
DOC_COLLECTION = os.getenv("MONGODB_COLLECTION", "marketing_texts")
SENT_COLLECTION = os.getenv("MONGODB_SENTENCE_COLLECTION", "marketing_sentence")
EMBED_COLLECTION = os.getenv("MONGODB_CRITERIA_COLLECTION", "marketing_policy")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is not set")
if not MONGODB_DB:
    raise RuntimeError("MONGODB_DB is not set")

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DB]

documents = db[DOC_COLLECTION]
sentence_analysis = db[SENT_COLLECTION]
embed_col = db[EMBED_COLLECTION]

print("MONGO URI: ", MONGODB_URI)
print("MONGO DB:", MONGODB_DB)
print("DOC COL :", documents.name)
print("SENT COL:", sentence_analysis.name)
print("EMBED COL: ", embed_col.name)


# RAG 저장함수
from datetime import datetime, timezone

def insert_embedding(
    *,
    content: str,
    embedding: list[float],
    metadata: dict | None = None,
):
    doc = {
        "content": content,
        "embedding": embedding,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc),
    }

    result = embed_col.insert_one(doc)
    return str(result.inserted_id)
