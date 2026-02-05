import chromadb
from sentence_transformers import SentenceTransformer

print(" Testing semantic search...")


print(" Creating client connection to ChromaDB...")
client_vectordb = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
collection = client_vectordb.get_collection("basic_rag_example")
print(" Client connection created!\n")

print(" Loading embeddings model...")
model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
print(" Embeddings model loaded!\n")

queries = [
    "¿Cómo puedo activar una tarjeta de crédito?",
    "¿Cómo puedo contratar una cuenta Básica BBVA?",
    "Cómo contrato la tarjeta oro?"
]

for query in queries:
    print(f" Query: {query}")
    query_embedding = model.encode(query).tolist()
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    for i, doc in enumerate(results['documents'][0]):
        score = results['distances'][0][i]
        print(f"  Result {i+1}: {doc} (Score: {score})")
    print("\n")
    print(" Semantic search test completed.")
