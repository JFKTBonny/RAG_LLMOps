"""Reusable RAG learning utilities for this project."""

from .config import DEFAULT_CHUNK_SIZE, DEFAULT_HANDBOOK_PATH, EMBEDDING_MODEL
from .data_loader import load_document_text, sliding_window_chunks
from .rag_pipeline import TechCorpRAGPipeline

__all__ = [
    "TechCorpRAGPipeline",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_HANDBOOK_PATH",
    "EMBEDDING_MODEL",
    "load_document_text",
    "sliding_window_chunks",
]
