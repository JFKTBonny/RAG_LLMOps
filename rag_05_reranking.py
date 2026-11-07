# rag_05_reranking.py — Reranking retrieved chunks for higher precision

from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
import ollama

# ── 1. DOCUMENTS ──────────────────────────────────────────────────────────
documents = [
    "The vehicle requires maintenance every 6 months to keep it running properly.",
    "All employees must submit vacation requests two weeks in advance via HR portal.",
    "The engineering team uses GitHub for version control and Jira for task tracking.",
    "Staff members should report any workplace injuries immediately to their supervisor.",
    "The automobile insurance policy covers collision damage and theft protection.",
    "Python and JavaScript are the primary programming languages used by developers.",
    "Workers are entitled to 15 days of paid leave per calendar year.",
    "The car engine should be inspected annually by a certified mechanic.",
    "Regular oil changes are essential for keeping your car engine healthy.",
    "Employees can work remotely up to 3 days per week with manager approval.",
]

# ── 2. SETUP RETRIEVAL (same as lesson 4) ────────────────────────────────
print("Setting up retrieval...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("reranking_demo")

embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

# ── 3. INITIAL RETRIEVAL — cast a wide net ────────────────────────────────


def initial_retrieval(question, n=6):
    """Get top n candidates from both dense and sparse."""
    # Dense
    q_embedding = embedder.encode(question).tolist()
    dense_results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n
    )["documents"][0]

    # Sparse
    tokenized_q = question.lower().split()
    scores = bm25.get_scores(tokenized_q)
    top_indices = np.argsort(scores)[::-1][:n]
    sparse_results = [documents[i] for i in top_indices]

    # Merge and deduplicate
    all_candidates = list(dict.fromkeys(dense_results + sparse_results))
    return all_candidates


# ── 4. RERANKER 1 — Cross encoder ────────────────────────────────────────
print("Loading cross-encoder reranker...")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cross_encoder_rerank(question, candidates, top_n=3):
    """
    Cross-encoder reads question + candidate TOGETHER.
    Much more precise than embedding similarity alone.
    """
    # Create pairs of [question, candidate] for the cross-encoder
    pairs = [[question, candidate] for candidate in candidates]

    # Score each pair
    scores = reranker.predict(pairs)

    # Sort candidates by score (highest first)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored[:top_n]]

# ── 5. RERANKER 2 — LLM based ────────────────────────────────────────────


def llm_rerank(question, candidates, top_n=3):
    """
    Ask the LLM itself to score relevance.
    Slowest but understands nuance best.
    """
    scored_candidates = []

    for candidate in candidates:
        prompt = f"""Score how relevant this document is to the question.
Return ONLY a number between 0 and 10. Nothing else.

Question: {question}
Document: {candidate}

Score:"""

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the number from response
        try:
            score = float(response["message"]["content"].strip())
        except:
            score = 0.0

        scored_candidates.append((candidate, score))

    # Sort by score
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_candidates[:top_n]]


# ── 6. COMPARE ALL APPROACHES ─────────────────────────────────────────────
question = "How often should I service my car?"

print(f"\n{'═'*60}")
print(f"Question: {question}")
print(f"{'─'*60}")

candidates = initial_retrieval(question)
print(f"\nInitial retrieval ({len(candidates)} candidates):")
for doc in candidates:
    print(f"  → {doc}")

print(f"\nAfter cross-encoder reranking (top 3):")
cross_results = cross_encoder_rerank(question, candidates)
for doc in cross_results:
    print(f"  → {doc}")

print(f"\nAfter LLM reranking (top 3):")
llm_results = llm_rerank(question, candidates)
for doc in llm_results:
    print(f"  → {doc}")
