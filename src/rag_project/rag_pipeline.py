from __future__ import annotations

from typing import List, Tuple

import numpy as np
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_STEP, EMBEDDING_MODEL, LLM_MODEL, resolve_data_path
from .data_loader import sliding_window_chunks


class TechCorpRAGPipeline:
    def __init__(self, source_path: str | None = None, collection_name: str = "techcorp_handbook"):
        self.source_path = resolve_data_path(source_path or "company_handbook.txt")
        self.collection_name = collection_name
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.chunks = self._load_chunks()
        self.collection = self._build_collection()
        self.bm25 = BM25Okapi([chunk.lower().split() for chunk in self.chunks])

    def _load_chunks(self) -> List[str]:
        text = self.source_path.read_text(encoding="utf-8")
        return sliding_window_chunks(text, chunk_size=DEFAULT_CHUNK_SIZE, step=DEFAULT_CHUNK_STEP)

    def _build_collection(self):
        import chromadb

        client = chromadb.Client()
        collection = client.create_collection(self.collection_name)
        embeddings = self.embedder.encode(self.chunks).tolist()
        collection.add(
            documents=self.chunks,
            embeddings=embeddings,
            ids=[f"chunk_{i}" for i in range(len(self.chunks))],
        )
        return collection

    def hybrid_retrieve(self, question: str, n: int = 6) -> List[str]:
        q_embedding = self.embedder.encode(question).tolist()
        dense_results = self.collection.query(query_embeddings=[q_embedding], n_results=n)["documents"][0]

        tokenized_q = question.lower().split()
        scores = self.bm25.get_scores(tokenized_q)
        top_indices = np.argsort(scores)[::-1][:n]
        sparse_results = [self.chunks[i] for i in top_indices]

        all_candidates = list(dict.fromkeys(dense_results + sparse_results))
        return all_candidates

    def rerank(self, question: str, candidates: List[str], top_n: int = 3) -> List[str]:
        pairs = [[question, candidate] for candidate in candidates]
        scores = self.reranker.predict(pairs)
        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored[:top_n]]

    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        context = "\n".join(context_chunks)
        prompt = f"""You are a helpful HR assistant for TechCorp.
Answer the question in a complete sentence that includes the topic of the question.
Use ONLY the information provided in the context below.
If the answer is not in the context, say "I don't have that information in the handbook."
Keep your answer concise — maximum 2 sentences.

Context:
{context}

Question: {question}

Answer:"""
        response = ollama.chat(
            model=LLM_MODEL,
            options={"temperature": 0.1},
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip()

    def quick_eval(self, question: str, answer: str, context_chunks: List[str]) -> float:
        context = "\n".join(context_chunks)
        prompt = f"""Score how faithful this answer is to the context.
Return ONLY a number between 0 and 1. Nothing else.
1.0 = fully grounded in context
0.0 = contains information not in context

Context: {context}
Answer: {answer}
Score:"""
        response = ollama.chat(
            model=LLM_MODEL,
            options={"temperature": 0.0},
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            return float(response["message"]["content"].strip())
        except Exception:
            return 0.0

    def ask(self, question: str) -> Tuple[str, float, List[str]]:
        candidates = self.hybrid_retrieve(question, n=6)
        top_chunks = self.rerank(question, candidates, top_n=3)
        answer = self.generate_answer(question, top_chunks)
        faith_score = self.quick_eval(question, answer, top_chunks)
        return answer, faith_score, top_chunks
