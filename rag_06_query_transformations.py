# rag_06_query_transformations.py — Improving queries before retrieval

from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ── 1. DOCUMENTS ──────────────────────────────────────────────────────────
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

# ── 2. SETUP VECTOR STORE ─────────────────────────────────────────────────
print("Setting up vector store...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("query_transform_demo")

embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)


def retrieve(question, n=2):
    """Basic dense retrieval."""
    q_embedding = embedder.encode(question).tolist()
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=n
    )
    return results["documents"][0]

# ── 3. TECHNIQUE 1 — MULTI-QUERY ──────────────────────────────────────────


def multi_query(original_question, n_variants=3):
    """Generate multiple versions of the question."""
    prompt = f"""Generate {n_variants} different ways to ask this question.
Each version should use different words but mean the same thing.
Return ONLY the questions, one per line, no numbering, no explanation.

Question: {original_question}

Versions:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    variants = response["message"]["content"].strip().split("\n")
    variants = [v.strip() for v in variants if v.strip()]
    return variants[:n_variants]


def multi_query_retrieval(original_question, n=2):
    """Retrieve using multiple query variants, merge results."""
    variants = multi_query(original_question)

    print(f"\n  Generated variants:")
    for v in variants:
        print(f"    • {v}")

    # Retrieve for each variant and merge
    all_results = []
    for variant in variants:
        results = retrieve(variant, n=n)
        all_results.extend(results)

    # Deduplicate while preserving order
    seen = set()
    unique_results = []
    for doc in all_results:
        if doc not in seen:
            seen.add(doc)
            unique_results.append(doc)

    return unique_results[:n*2]

# ── 4. TECHNIQUE 2 — HyDE ─────────────────────────────────────────────────


def hyde_retrieval(question, n=2):
    """Generate hypothetical answer, embed that instead of the question."""
    prompt = f"""Write a short factual paragraph that would perfectly answer this question.
Write it as if it comes from a company document.
Keep it under 100 words.

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    hypothetical_answer = response["message"]["content"].strip()
    print(f"\n  Hypothetical answer generated:")
    print(f"  '{hypothetical_answer[:150]}...'")

    # Embed the hypothetical answer instead of the question
    return retrieve(hypothetical_answer, n=n)

# ── 5. TECHNIQUE 3 — STEP-BACK ────────────────────────────────────────────


def step_back_retrieval(specific_question, n=2):
    """Broaden the question to retrieve more context."""
    prompt = f"""This question is very specific. Generate a broader, more general
version of it that would help find background context.
Return ONLY the broader question, nothing else.

Specific question: {specific_question}

Broader question:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )

    broader_question = response["message"]["content"].strip()
    print(f"\n  Step-back question: '{broader_question}'")

    return retrieve(broader_question, n=n)


# ── 6. COMPARE ALL THREE ──────────────────────────────────────────────────
questions = [
    "What's the WFH policy?",                    # acronym test
    "When exactly does my perf review happen?",   # specific question
]

for question in questions:
    print(f"\n{'═'*60}")
    print(f"Original question: {question}")

    print(f"\n── Baseline (no transformation):")
    for doc in retrieve(question):
        print(f"  → {doc}")

    print(f"\n── Multi-query retrieval:")
    for doc in multi_query_retrieval(question):
        print(f"  → {doc}")

    print(f"\n── HyDE retrieval:")
    for doc in hyde_retrieval(question):
        print(f"  → {doc}")

    print(f"\n── Step-back retrieval:")
    for doc in step_back_retrieval(question):
        print(f"  → {doc}")
