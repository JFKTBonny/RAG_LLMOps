# RAG MLOps Learning Project

This repository is a small, hands-on RAG (retrieval-augmented generation) learning project. It walks through the core ideas behind embeddings, vector stores, chunking, retrieval, reranking, query transformations, evaluation, and a full end-to-end pipeline.

## Project goals

- Learn the fundamentals of RAG in a simple, understandable way.
- Experiment with local embeddings and retrieval logic.
- Work with real documents and policy-style content.
- Gain a practical understanding of the full RAG pipeline.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── requirements-freeze.txt
├── company_handbook.txt
├── sdlc_handbook.txt
├── notes.txt
├── company_doc.pdf
├── test_embeddings.py
├── rag_01_embeddings_and_vectorstore.py
├── rag_02_generation_with_llm.py
├── rag_03_chunking_strategies.py
├── rag_04_retrieval_techniques.py
├── rag_05_reranking.py
├── rag_06_query_transformations.py
├── rag_07_evaluation.py
├── rag_08_full_pipeline.py
├── data/
│   └── (optional data files or future corpus assets)
├── src/
│   └── rag_project/
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       └── rag_pipeline.py
└── .gitignore
```

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the lesson scripts:

```bash
python rag_01_embeddings_and_vectorstore.py
python rag_08_full_pipeline.py
```

## Notes on the scripts

The numbered `rag_*.py` scripts are tutorial examples. They intentionally stay small and readable so you can follow each concept in isolation.

The `src/rag_project` package contains reusable logic for a cleaner, more production-friendly version of the full pipeline.

## Recommended next improvements

- Add a proper test suite for retrieval quality and answer grounding.
- Split the project into a data ingestion layer, indexing layer, and evaluation layer.
- Replace hardcoded file names with configuration-driven settings.
- Add a web UI or API wrapper.
- Add Docker support for repeatable local environments.

## Environment notes

The project uses local open-source models and libraries where possible. Some examples call `ollama`, so if you want the LLM generation examples to work, make sure Ollama is installed and the model is available locally.

Example:

```bash
ollama pull llama3.2
```
