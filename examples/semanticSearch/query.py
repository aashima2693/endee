import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load embeddings
with open("embeddings.json", "r") as f:
    data = json.load(f)

texts = [item["text"] for item in data]
vectors = np.array([item["embedding"] for item in data])

query = input("Ask a question: ")
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, vectors)[0]
top_indices = similarities.argsort()[-3:][::-1]

print("\nTop Results:\n")

for idx in top_indices:
    print("Source:", data[idx]["source"])
    print(texts[idx])
    print()