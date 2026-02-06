import chromadb
import os
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

print(" Testing answer generation...")

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

query = "¿Cómo puedo contratar una cuenta Básica BBVA?"

query_embedding = model.encode(query).tolist()
    
result = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)
 
context = "\n\n".join(result['documents'][0])

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
    answer_text = response.output_text

except Exception as e:
    print(f"Error generating answer: {e}")
    answer_text = f"Error generating answer: {e}"

print(" Generated Answer:")
print(answer_text)
print("\n Answer generation test completed.")
