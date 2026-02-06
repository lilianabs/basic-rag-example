import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

def get_client_vectordb():
    client_vectordb = chromadb.PersistentClient(
        path="./chroma_db",
        settings=chromadb.config.Settings(anonymized_telemetry=False)
    )
    collection = client_vectordb.get_collection("basic_rag_example")
    return collection

def get_embeddings_model():
    model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
    return model

def get_context(collection, model, query, n_results=3):
    query_embedding = model.encode(query).tolist()
    
    context = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return "\n\n".join(context['documents'][0])

def generate_answer(query, context):
    prompt = '''
    Eres un asistente útil que proporciona respuestas precisas basadas en el contexto proporcionado.
    Usa el siguiente contexto {context} para responder a la pregunta {query}.'''
    
    formatted_prompt = prompt.format(context=context, query=query)
    
    client = OpenAI()
    
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=formatted_prompt
        )
        
        output_text = ""
        for item in response.output:
            if hasattr(item, "content"):
                for content in item.content:
                    if hasattr(content, "text"):
                        output_text += content.text
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return f"Error generating answer: {e}"
    
    return output_text
    
def run_rag_pipeline(query: str):
    print(" Running RAG pipeline with ChromaDB and OpenAI...\n")
    
    print(" Setting up RAG pipeline...")
    print(" Creating client connection to ChromaDB...")
    collection = get_client_vectordb()
    print(" Client connection created!\n")
    print(" Loading embeddings model...")
    model = get_embeddings_model()
    print(" Embeddings model loaded!\n")
    print(" RAG pipeline setup completed!\n")
    
    print(f'Retrieving context for query: {query}')
    context = get_context(collection, model, query)
    print(f"Retrieved context: {context}")
    
    

if __name__ == "__main__":
    query = "¿Cómo puedo contratar una cuenta Básica BBVA?"
    
    run_rag_pipeline(query)