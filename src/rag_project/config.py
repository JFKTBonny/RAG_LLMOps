from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"
DEFAULT_CHUNK_SIZE = 80
DEFAULT_CHUNK_STEP = 40


def resolve_data_path(filename: str) -> Path:
    """Return the preferred project path for a document file.

    Prefer the `data/` folder, but allow fallback to the repo root for legacy files.
    """
    candidates = [DATA_DIR / filename, PROJECT_ROOT / filename]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_HANDBOOK_PATH = resolve_data_path("company_handbook.txt")
DEFAULT_PDF_PATH = resolve_data_path("company_doc.pdf")
