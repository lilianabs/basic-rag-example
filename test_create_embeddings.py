from sentence_transformers import SentenceTransformer
import numpy as np

print(" Loading Embeddings Model...")
model = SentenceTransformer('jaimevera1107/all-MiniLM-L6-v2-similarity-es')
print(" Embeddings model loaded!\n")

sentences = [
    "Contratar tarjeta de crédito",
    "Abrir una cuenta de ahorros",
    "Solicitar tarjeta de credito adicional",
]

print(" Creating embeddings...")
embeddings = model.encode(sentences)
print(f" Created {len(embeddings)} vectors of {len(embeddings[0])} dimensions each!\n")

# Calculate semantic similarities
sim_1_2 = np.dot(embeddings[0], embeddings[1])
sim_1_3 = np.dot(embeddings[0], embeddings[2])
print(f" Similarity between sentence 1 and 2: {sim_1_2}")
print(f" Similarity between sentence 1 and 3: {sim_1_3}")
