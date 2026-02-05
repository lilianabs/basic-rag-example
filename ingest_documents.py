import os
import chromadb
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

print("  Ingesting documents into vector database ...")
print("="*50)

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

print(" Loading documents...")
path = Path('data/raw/FAQs_Evaluacion_LLMs.xlsx')
df = pd.read_excel(path)
df = df.rename(columns={'Title': 'question', 'Final response': 'answer'})
print(f" Loaded {len(df)} documents!\n")

print(" Creating embeddings for documents...")
doc_count = 0
for index, row in df.iterrows():
    chunk = row['answer']
    question = row['question']
    
    embedding = model.encode(chunk).tolist()
    
    collection.add(
        ids=[str(index)],
        embeddings=[embedding],
        documents=[chunk],
        metadatas={"question": question}
    )
    doc_count += 1
    print(f"  Document {index} ingested.")
print(f"\n Created embeddings and ingested {doc_count} documents into vector database!\n")

