import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

DOC_FOLDER = "docs"
OUTPUT_FILE = "embeddings.json"

documents = []
sources = []
for file in os.listdir(DOC_FOLDER):
    if file.endswith(".txt"):
        with open(os.path.join(DOC_FOLDER, file), "r") as f:
            text = f.read()

            sentences = text.split(".")
            for sentence in sentences:
                sentence = sentence.strip()

                if sentence:
                    documents.append(sentence)
                    sources.append(file)

print("Total chunks:", len(documents))
embeddings = model.encode(documents)

data = []

for i in range(len(documents)):
    data.append({
        "text": documents[i],
        "embedding": embeddings[i].tolist(),
        "source": sources[i]
    })
with open(OUTPUT_FILE, "w") as f:
    json.dump(data, f)

print("Embeddings saved to embeddings.json")