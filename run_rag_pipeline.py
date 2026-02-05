import chromadb
from sentence_transformers import SentenceTransformer
import openai

def get_client_vectordb():
    print(" Creating client connection to ChromaDB...")
    client_vectordb = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection = client_vectordb.get_collection("basic_rag_example")
    print(" Client connection created!\n")
    return collection

def get_embeddings_model():
    print(" Loading embeddings model...")
    model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
    print(" Embeddings model loaded!\n")
    return model

def get_context(collection, model, query, n_results=3):
    print(f'Retrieving context for query: {query}')
    query_embedding = model.encode(query).tolist()
    
    context = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return "\n\n".join(context['documents'][0])
    
def run_rag_pipeline(query: str):
    print(" Running RAG pipeline with ChromaDB and OpenAI...\n")
    
    print(" Setting up RAG pipeline...")
    collection = get_client_vectordb()
    model = get_embeddings_model()
    print(" RAG pipeline setup completed!\n")
    
    context = get_context(collection, model, query)
    print(f"Retrieved context: {context}")
    
    

if __name__ == "__main__":
    query = "¿Cómo puedo contratar una cuenta Básica BBVA?"
    
    run_rag_pipeline(query)