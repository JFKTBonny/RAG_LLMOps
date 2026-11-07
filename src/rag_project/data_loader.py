from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from .config import resolve_data_path


def load_document_text(path: str | Path | None = None) -> str:
    """Load plain text from a .txt or .pdf file."""
    target = Path(path) if path is not None else resolve_data_path("company_handbook.txt")
    if target.suffix.lower() == ".pdf":
        return extract_pdf_text(target)
    return target.read_text(encoding="utf-8")


def extract_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return full_text


def sliding_window_chunks(text: str, chunk_size: int = 80, step: int = 40) -> List[str]:
    """Chunk text using a sliding window over words."""
    words = text.split()
    chunks: List[str] = []
    for i in range(0, len(words), step):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) < 20:
            break
        chunks.append(" ".join(chunk_words))
    return chunks
