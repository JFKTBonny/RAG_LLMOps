

# lesson2.py — Minimal RAG from scratch

from sentence_transformers import SentenceTransformer
import chromadb

# ── 1. OUR FAKE "DOCUMENTS" (we'll use real PDFs later) ──────────────────
documents = [
    "The onboarding process takes 3 days and includes HR, IT setup, and team introduction.",
    "All employees must complete security training within the first week.",
    "The engineering team uses GitHub for version control and Jira for task tracking.",
    "Vacation requests must be submitted at least 2 weeks in advance via the HR portal.",
    "The company provides a $500 annual budget for learning and development.",
]

# ── 2. LOAD THE EMBEDDING MODEL (runs locally, no API needed) ─────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── 3. CREATE A LOCAL VECTOR DATABASE ────────────────────────────────────
client = chromadb.Client()
collection = client.create_collection("company_docs")

# ── 4. EMBED AND STORE DOCUMENTS ─────────────────────────────────────────
print("Embedding documents...")
embeddings = embedder.encode(documents).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

print(f"Stored {len(documents)} documents in vector DB\n")

# ── 5. QUERY — THIS IS THE RAG PART ──────────────────────────────────────
question = "How many days does onboarding take?"

print(f"Question: {question}")

# Convert question to embedding and search
question_embedding = embedder.encode(question).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2  # retrieve top 2 most relevant chunks
)

print("\nRetrieved chunks:")
for doc in results["documents"][0]:
    print(f"  → {doc}")
