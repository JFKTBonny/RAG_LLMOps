
# rag_04_retrieval_techniques.py — Dense vs Sparse vs Hybrid retrieval

from sentence_transformers import SentenceTransformer
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np

# ── 1. DOCUMENTS ──────────────────────────────────────────────────────────
# Notice we have vocabulary mismatches between docs and questions
documents = [
    "The vehicle requires maintenance every 6 months to keep it running properly.",
    "All employees must submit vacation requests two weeks in advance via HR portal.",
    "The engineering team uses GitHub for version control and Jira for task tracking.",
    "Staff members should report any workplace injuries immediately to their supervisor.",
    "The automobile insurance policy covers collision damage and theft protection.",
    "Python and JavaScript are the primary programming languages used by developers.",
    "Workers are entitled to 15 days of paid leave per calendar year.",
    "The car engine should be inspected annually by a certified mechanic.",
]

# ── 2. SETUP EMBEDDING MODEL AND VECTOR STORE ─────────────────────────────
print("Setting up dense retrieval (embeddings)...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("retrieval_demo")

embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# ── 3. SETUP BM25 FOR SPARSE RETRIEVAL ───────────────────────────────────
print("Setting up sparse retrieval (BM25)...")

# BM25 works on tokenized text (list of words per document)
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ── 4. DEFINE OUR THREE RETRIEVAL FUNCTIONS ───────────────────────────────


def dense_retrieval(question, n=3):
    """Embedding-based semantic search."""
    question_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n
    )
    return results["documents"][0]


def sparse_retrieval(question, n=3):
    """BM25 keyword-based search."""
    tokenized_question = question.lower().split()
    scores = bm25.get_scores(tokenized_question)
    # Get indices of top n scores
    top_indices = np.argsort(scores)[::-1][:n]
    return [documents[i] for i in top_indices]


def hybrid_retrieval(question, n=3):
    """
    Combine dense and sparse scores.
    Strategy: get top results from both, merge and deduplicate,
    return the ones that appear in both (highest confidence).
    """
    dense_results = set(dense_retrieval(question, n=5))
    sparse_results = set(sparse_retrieval(question, n=5))

    # Documents found by BOTH methods = highest confidence
    both = dense_results & sparse_results

    # Documents found by only one method
    only_dense = dense_results - sparse_results
    only_sparse = sparse_results - dense_results

    # Combine: both first, then fill remaining slots
    combined = list(both) + list(only_dense) + list(only_sparse)
    return combined[:n]

# ── 5. TEST WITH TWO QUESTIONS ─────────────────────────────────────────────


questions = [
    "How often should I service my car?",      # vocabulary mismatch test
    "How many days off do employees get?",      # synonym test
]

for question in questions:
    print(f"\n{'═'*60}")
    print(f"Question: {question}")
    print(f"{'─'*60}")

    print("\n DENSE retrieval (embeddings):")
    for doc in dense_retrieval(question):
        print(f"  → {doc}")

    print("\n SPARSE retrieval (BM25):")
    for doc in sparse_retrieval(question):
        print(f"  → {doc}")

    print("\n HYBRID retrieval (combined):")
    for doc in hybrid_retrieval(question):
        print(f"  → {doc}")
