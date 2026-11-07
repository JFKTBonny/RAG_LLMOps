# rag_07_evaluation.py — Measuring RAG quality objectively

from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import numpy as np

# ── 1. SETUP ──────────────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "The vehicle requires maintenance every 6 months to keep it running properly.",
    "All employees must submit vacation requests two weeks in advance via HR portal.",
    "The engineering team uses GitHub for version control and Jira for task tracking.",
    "Staff members should report any workplace injuries immediately to their supervisor.",
    "Regular oil changes are essential for keeping your car engine healthy.",
    "Employees can work remotely up to 3 days per week with manager approval.",
    "The company provides health insurance covering medical, dental and vision.",
    "New hires must complete onboarding documentation within the first 3 days.",
    "The IT department handles all software installation requests within 24 hours.",
    "Performance reviews are conducted twice a year in June and December.",
]

client = chromadb.Client()
collection = client.create_collection("eval_demo")
embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


def retrieve(question, n=3):
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n
    )
    return results["documents"][0]


def generate_answer(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}
Answer:"""
    response = ollama.chat(
        model="llama3.2",
        # options={"temperature": 0.1},
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()

# ── 2. METRIC 1 — CONTEXT PRECISION ──────────────────────────────────────


def context_precision(question, retrieved_chunks, relevant_chunks):
    relevant_set = set(relevant_chunks)
    hits = sum(1 for chunk in retrieved_chunks if chunk in relevant_set)
    precision = hits / len(retrieved_chunks) if retrieved_chunks else 0
    return precision

# ── 3. METRIC 2 — CONTEXT RECALL ─────────────────────────────────────────


def context_recall(retrieved_chunks, relevant_chunks):
    relevant_set = set(relevant_chunks)
    hits = sum(1 for chunk in relevant_chunks if chunk in retrieved_chunks)
    recall = hits / len(relevant_chunks) if relevant_chunks else 0
    return recall

# ── 4. METRIC 3 — FAITHFULNESS ────────────────────────────────────────────


def faithfulness_score(answer, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""Given this context and answer, score how faithful the answer
is to the context on a scale of 0 to 1.
1.0 = every claim in the answer is directly supported by the context.
0.0 = the answer contains claims not found in the context (hallucination).
Return ONLY a number between 0 and 1. Nothing else.

Context:
{context}

Answer: {answer}

Faithfulness score:"""

    response = ollama.chat(
        model="llama3.2",
        # options={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}]
    )
    try:
        return float(response["message"]["content"].strip())
    except:
        return 0.0

# ── 5. METRIC 4 — ANSWER RELEVANCE ───────────────────────────────────────


def answer_relevance_score(original_question, answer):
    prompt = f"""Given this answer, what question was it trying to answer?
Return ONLY the question, nothing else.

Answer: {answer}

Question:"""

    response = ollama.chat(
        model="llama3.2",
        # options={"temperature": 0.0},
        messages=[{"role": "user", "content": prompt}]
    )

    regenerated_question = response["message"]["content"].strip()

    orig_embedding = embedder.encode(original_question)
    regen_embedding = embedder.encode(regenerated_question)

    similarity = np.dot(orig_embedding, regen_embedding) / (
        np.linalg.norm(orig_embedding) * np.linalg.norm(regen_embedding)
    )

    return float(similarity), regenerated_question


# ── 6. RUN FULL EVALUATION ────────────────────────────────────────────────
test_cases = [
    {
        "question": "How often should I service my car?",
        "relevant_chunks": [
            "The vehicle requires maintenance every 6 months to keep it running properly.",
            "Regular oil changes are essential for keeping your car engine healthy.",
        ]
    },
    {
        "question": "What are the remote work options?",
        "relevant_chunks": [
            "Employees can work remotely up to 3 days per week with manager approval.",
        ]
    },
    {
        "question": "When are performance reviews?",
        "relevant_chunks": [
            "Performance reviews are conducted twice a year in June and December.",
        ]
    },
]

print("Running RAG evaluation...\n")
print("═" * 60)

all_precision = []
all_recall = []
all_faithfulness = []
all_relevance = []

for test in test_cases:
    question = test["question"]
    relevant_chunks = test["relevant_chunks"]

    print(f"\nQuestion: {question}")
    print("─" * 60)

    retrieved = retrieve(question, n=3)
    answer = generate_answer(question, retrieved)

    print(f"Answer: {answer}")

    precision = context_precision(question, retrieved, relevant_chunks)
    recall = context_recall(retrieved, relevant_chunks)
    faith = faithfulness_score(answer, retrieved)
    relevance, regenerated_q = answer_relevance_score(question, answer)

    all_precision.append(precision)
    all_recall.append(recall)
    all_faithfulness.append(faith)
    all_relevance.append(relevance)

    print(f"\nMetrics:")
    print(f"  Context precision:  {precision:.2f}")
    print(f"  Context recall:     {recall:.2f}")
    print(f"  Faithfulness:       {faith:.2f}")
    print(f"  Answer relevance:   {relevance:.2f}")
    print(f"  (Regenerated Q:     '{regenerated_q}')")

print(f"\n{'═'*60}")
print("OVERALL RAG SYSTEM SCORES:")
print(f"  Avg context precision:  {np.mean(all_precision):.2f}")
print(f"  Avg context recall:     {np.mean(all_recall):.2f}")
print(f"  Avg faithfulness:       {np.mean(all_faithfulness):.2f}")
print(f"  Avg answer relevance:   {np.mean(all_relevance):.2f}")
print(
    f"\n  Overall RAG score:      {np.mean(all_precision + all_recall + all_faithfulness + all_relevance):.2f}")
