# rag_02_generation_with_llm.py — Adding LLM generation to complete the RAG loop

from sentence_transformers import SentenceTransformer
import chromadb
import ollama

# ── 1. DOCUMENTS ──────────────────────────────────────────────────────────
documents = [
    "The onboarding process takes 3 days and includes HR, IT setup, and team introduction.",
    "All employees must complete security training within the first week.",
    "The engineering team uses GitHub for version control and Jira for task tracking.",
    "Vacation requests must be submitted at least 2 weeks in advance via the HR portal.",
    "The company provides a $500 annual budget for learning and development.",
]

# ── 2. EMBED AND STORE (same as lesson 1) ────────────────────────────────
print("Setting up vector store...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.create_collection("company_docs")

embeddings = embedder.encode(documents).tolist()
collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=[f"doc_{i}" for i in range(len(documents))]
)

# ── 3. RETRIEVAL ──────────────────────────────────────────────────────────
question = "how many employees has the compagny?"

print(f"\nQuestion: {question}")

question_embedding = embedder.encode(question).tolist()
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)

retrieved_chunks = results["documents"][0]

print("\nRetrieved chunks:")
for chunk in retrieved_chunks:
    print(f"  → {chunk}")

# ── 4. GENERATION — THE NEW PART ─────────────────────────────────────────

# We build a prompt that contains:
#   - the retrieved context
#   - the user question
# This is called a RAG prompt

context = "\n".join(retrieved_chunks)

prompt = f"""You are a helpful company assistant.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""

print("\nSending to LLM...")
response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": prompt}]
)

answer = response["message"]["content"]

print(f"\nAnswer: {answer}")
