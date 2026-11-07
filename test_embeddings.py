# save as test_embeddings.py
from sentence_transformers import SentenceTransformer # sentence-transformers library for embeddings is used for transforming text into vector representations..
import numpy as np

embedder = SentenceTransformer("all-MiniLM-L6-v2") # Initialize the sentence transformer with a pre-trained model

words = ["car", "vehicle", "automobile", "banana", "truck", "python"]
embeddings = embedder.encode(words)

# Compare "car" to every other word
car_embedding = embeddings[0]

print("Similarity to 'car':")
for i, word in enumerate(words):
    # cosine similarity — 1.0 = identical, 0.0 = unrelated, -1.0 = opposite
    similarity = np.dot(car_embedding, embeddings[i]) / (
        np.linalg.norm(car_embedding) * np.linalg.norm(embeddings[i])
    )
    print(f"  {word:15} → {similarity:.4f}")
