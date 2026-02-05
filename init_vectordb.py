import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="basic_rag_example",
    metadata={"hnsw:space": "cosine"}
)

print(f" Collection created: {collection.name}")
print(f" Elements in the collection: {collection.count()}")
