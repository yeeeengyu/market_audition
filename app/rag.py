import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_one(text: str) -> list[float]:
    text = text.strip()
    if not text:
        return []
    resp = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        input=text
    )
    return resp.data[0].embedding

def retrieve(criteria_col, query: str, doc_type: str, k: int = 3):
    qvec = embed_one(query)
    if not qvec:
        return []

    pipeline = [
        {
            "$vectorSearch": {
                "index": os.getenv("VECTOR_INDEX_NAME", "criteria_vector_index"),
                "path": "embedding",
                "queryVector": qvec,
                "numCandidates": 200,
                "limit": k,
                "filter": {"type": doc_type}
            }
        },
        {
            "$project": {
                "_id": 1,
                "type": 1,
                "title": 1,
                "content": 1,
                "source": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    return list(criteria_col.aggregate(pipeline))
