# AGENTS.md

This file provides guidance to code agents when working with code in this repository.

## Project

Minimal local-first RAG pipeline for arXiv cs.CV (Computer Vision) papers. Fetches papers from arXiv, parses PDFs via GROBID, chunks and embeds text, stores in Qdrant + SQLite FTS5, and answers queries using hybrid retrieval with MLX-based generation.

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Start Qdrant vector database
docker compose up -d

# Run CLI commands
uv run cv-rag ingest --limit 10
uv run cv-rag ingest-ids 2104.00680 1911.11763
uv run cv-rag query "vision transformers"
uv run cv-rag answer "vision transformers" --model <mlx-model-path>
uv run cv-rag doctor
uv run cv-rag curate --refresh-days 30

# Web UI (install web deps, start server, open browser)
uv sync --extra web
uv run cv-rag serve                     # http://127.0.0.1:8000
uv run cv-rag serve --reload            # dev mode with auto-reload

# Frontend development (separate terminal)
cd web && npm install && npm run dev    # http://localhost:5173 (proxies /api to :8000)
cd web && npm run build                 # builds to cv_rag/api/static/

# Run tests
uv run pytest
uv run pytest tests/test_answer.py::test_build_strict_answer_prompt_includes_sources_and_rules

# Lint
uv run ruff check cv_rag/ tests/
uv run ruff check --fix cv_rag/ tests/  # auto-fix

# Verbose logging (any command)
uv run cv-rag -v ingest --limit 5
```

## Architecture

### Module Layout

```
cv_rag/
    cli.py            # Thin Typer CLI wiring (~700 lines), delegates to service modules
    config.py         # Pydantic Settings, env vars prefixed CV_RAG_
    exceptions.py     # Exception hierarchy: CvRagError → IngestError, RetrievalError, GenerationError, CitationValidationError
    http_retry.py     # Shared HTTP retry with exponential backoff (used by arxiv_sync, grobid_client)
    llm.py            # MLX subprocess wrapper (mlx_generate), raises GenerationError
    answer.py         # Answer orchestration: prompts, citation validation, comparison detection, chunk merging
    ingest.py         # IngestPipeline: download → parse → chunk → embed → store
    arxiv_sync.py     # ArXiv API feed fetching + PDF download
    grobid_client.py  # GROBID PDF→TEI XML
    tei_extract.py    # TEI XML → Section list
    chunking.py       # Sliding window chunker (1200 chars, 200 overlap)
    embeddings.py     # Ollama embedding client
    retrieve.py       # HybridRetriever: RRF fusion of Qdrant vector + SQLite FTS5 BM25
    qdrant_store.py   # Qdrant vector DB wrapper
    sqlite_store.py   # SQLite FTS5 store for BM25 + metadata
    api/              # FastAPI web UI backend (optional dep: uv sync --extra web)
        app.py        # App factory, CORS, lifespan, static mount
        deps.py       # FastAPI dependency injection
        schemas.py    # Pydantic request/response models
        streaming.py  # SSE helpers + mlx_generate_stream()
        routers/      # health, stats, papers, search, answer endpoints
web/                  # React + TypeScript + Tailwind frontend (Vite)
    src/
        pages/        # ChatPage, PapersPage, PaperDetailPage, StatsPage, HealthPage
        components/   # layout/, chat/, papers/, stats/, health/
        hooks/        # useChat, useHealth, useStats, usePapers
        api/          # TypeScript API client + SSE streaming
```

### Ingestion Pipeline
ArXiv API → `arxiv_sync.py` (fetch metadata + PDFs) → `grobid_client.py` (PDF→TEI XML) → `tei_extract.py` (extract sections) → `chunking.py` (sliding window) → embeddings via Ollama (`embeddings.py`) → stored in both Qdrant (`qdrant_store.py`) and SQLite FTS5 (`sqlite_store.py`). Orchestrated by `IngestPipeline` in `ingest.py`.

### Hybrid Retrieval
`retrieve.py` implements `HybridRetriever` that merges vector search (Qdrant, cosine) with keyword search (SQLite FTS5, BM25) using Reciprocal Rank Fusion (RRF). Applies per-document chunk quotas, section boosting, and deduplication by (arxiv_id, section_title, chunk_id).

### Relevance & Answer Safeguards
- **Relevance guard**: Extracts rare query terms (5+ chars, digits); raises `NoRelevantSourcesError` if no term overlap and vector scores below threshold (`relevance_vector_threshold` in Settings, default 0.45).
- **Comparison guard**: Detects comparison queries via regex; refuses if fewer than 2 sources from each of the top 2 papers.
- **Citation enforcement**: Checks generated answers for `[S1][S3]`-style citations; re-prompts if missing. Validation logic in `answer.py`.

### Key Patterns
- **HTTP retry**: All external HTTP calls use `http_retry.http_request_with_retry()` with exponential backoff. GROBID uses `prepare_kwargs` callback to re-open file handles per attempt.
- **Exception hierarchy**: `exceptions.py` defines `CvRagError` base. Modules raise specific subclasses; CLI catches and converts to `typer.Exit`.
- **Logging**: All service modules use `logging.getLogger(__name__)`. Enable with `cv-rag -v`.

## Configuration

All settings in `config.py` via environment variables prefixed `CV_RAG_`. Key ones:
- `CV_RAG_QDRANT_URL` (default: localhost:6333), `CV_RAG_GROBID_URL` (localhost:8070), `CV_RAG_OLLAMA_URL` (localhost:11434)
- `CV_RAG_CHUNK_MAX_CHARS` (1200), `CV_RAG_CHUNK_OVERLAP` (200)
- Storage paths: `CV_RAG_DATA_DIR`, `CV_RAG_PDF_DIR`, `CV_RAG_TEI_DIR`, `CV_RAG_METADATA_DIR`

## External Services

Qdrant (vector DB), GROBID (PDF parsing), and Ollama (embeddings + LLM) must be running. Use `cv-rag doctor` to verify connectivity.

## Tooling

- **Linter**: ruff (configured in `pyproject.toml`). `B008` ignored for Typer defaults.
- **Type checker**: mypy (configured in `pyproject.toml`, strict mode).
- **Tests**: pytest. All tests are offline (monkeypatched). Run with `uv run pytest -q`.
