from config import settings
import pinecone

pc = pinecone.Pinecone(api_key=settings.PINECONE_API_KEY)
index = pc.Index(settings.PINECONE_INDEX)

def upsert_vectors(vectors):
    BATCH = 100
    for i in range(0, len(vectors), BATCH):
        index.upsert(vectors=vectors[i:i+BATCH])