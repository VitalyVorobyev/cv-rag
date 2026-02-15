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
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl
uv run cv-rag ingest-ids 2104.00680 1911.11763 --no-skip-ingested
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl --no-skip-ingested
uv run cv-rag query "vision transformers"
uv run cv-rag answer "vision transformers" --mode auto --model <mlx-model-path>
uv run cv-rag seed-awesome --sources data/curation/awesome_sources.txt --out-dir data/curation
uv run cv-rag seed awesome --sources data/curation/awesome_sources.txt --out-dir data/curation
uv run cv-rag seed visionbib --sources data/curation/visionbib_sources.txt --out-dir data/curation/visionbib
uv run cv-rag resolve-dois --dois data/curation/tierA_dois.txt --out-dir data/curation
uv run cv-rag resolve-dois --dois data/curation/tierA_dois_visionbib.txt --out-dir data/curation/visionbib --tierA-arxiv-from-openalex data/curation/tierA_arxiv_openalex_visionbib.txt
uv run cv-rag doctor
uv run cv-rag curate --refresh-days 30

# Web UI (install web deps, start server, open browser)
uv sync --extra web
uv run cv-rag serve                     # http://127.0.0.1:8000
uv run cv-rag serve --reload            # dev mode with auto-reload

# Frontend development (separate terminal)
cd web && npm install && npm run dev    # http://localhost:5173 (proxies /api to :8000)
cd web && npm run build                 # builds to cv_rag/interfaces/api/static/

# Run tests
uv run pytest
uv run pytest tests/test_answer.py::test_build_strict_answer_prompt_includes_sources_and_rules

# Lint
uv run ruff check cv_rag/ tests/
uv run ruff check --fix cv_rag/ tests/  # auto-fix

# Verbose logging (any command)
uv run cv-rag -v ingest --limit 5
```

`ingest`, `ingest-ids`, and `ingest-jsonl` all default to exact-version dedupe (`--skip-ingested`) and can be overridden with `--no-skip-ingested`. For unversioned explicit IDs, cv-rag attempts version resolution first and continues best-effort with a warning if unresolved.
`seed visionbib` reads a VisionBib prefix + page range spec and emits dedicated DOI/PDF/arXiv seed artifacts.
`resolve-dois` can optionally export recovered arXiv IDs from OpenAlex via `--tierA-arxiv-from-openalex`.

## Architecture

### Module Layout

```
cv_rag/
    app/bootstrap.py              # shared runtime construction (settings + stores + retriever)
    shared/                       # settings, errors, shared HTTP/backoff helpers
    storage/                      # sqlite + qdrant adapters, storage DTOs
    ingest/                       # arXiv client, TEI/chunking pipeline, dedupe, ingest service
    retrieval/                    # hybrid retrieval + relevance filtering helpers
    answer/                       # routing, prompts, citation checks, answer service, MLX runner
    curation/                     # semantic-scholar enrichment and tiering
    seeding/                      # awesome-list and OpenAlex DOI seeding/resolution
    interfaces/
        cli/app.py                # Typer CLI entrypoint
        cli/commands/             # command handlers by domain
        api/                      # FastAPI app, routers, deps, schemas, static SPA mount
    embeddings.py                 # Ollama embedding client
web/                              # React + TypeScript + Tailwind frontend (Vite)
```

### Ingestion Pipeline
ArXiv API (`ingest/arxiv_client.py`) → GROBID (`ingest/grobid_client.py`) → TEI extraction (`ingest/tei_extract.py`) → chunking (`ingest/chunking.py`) → embeddings (`embeddings.py`) → storage (`storage/qdrant.py`, `storage/sqlite.py`). Orchestrated by `ingest/pdf_pipeline.py` and `ingest/service.py`.

### Hybrid Retrieval
`retrieval/hybrid.py` implements `HybridRetriever` that merges vector search (Qdrant, cosine) with keyword search (SQLite FTS5, BM25) using Reciprocal Rank Fusion (RRF). Applies per-document chunk quotas, section boosting, and deduplication by `(arxiv_id, section_title, chunk_id)`.

### Relevance & Answer Safeguards
- **Relevance guard**: Extracts rare query terms (5+ chars, digits); raises `NoRelevantSourcesError` if no term overlap and vector scores below threshold (`relevance_vector_threshold` in Settings, default 0.45).
- **Answer routing**: `answer` uses a cheap prelim retrieval pass, then routes to `single|compare|survey|implement|evidence` via rules/LLM/hybrid (`answer/routing.py`).
- **Comparison guard**: Applied when routed mode is compare; refuses if fewer than 2 sources from each of the top 2 papers.
- **Citation enforcement**: Generated answers must include `[S#]` citations; CLI/API share `answer/service.py` + `answer/citations.py`.

### Key Patterns
- **HTTP retry**: External HTTP calls use `shared/http.py` with exponential backoff. GROBID uses `prepare_kwargs` callbacks to re-open file handles per attempt.
- **Exception hierarchy**: `shared/errors.py` defines `CvRagError` base. Modules raise specific subclasses; CLI catches and converts to `typer.Exit`.
- **Logging**: All service modules use `logging.getLogger(__name__)`. Enable with `cv-rag -v`.

## Configuration

All settings are in `shared/settings.py` via environment variables prefixed `CV_RAG_`. Key ones:
- `CV_RAG_QDRANT_URL` (default: localhost:6333), `CV_RAG_GROBID_URL` (localhost:8070), `CV_RAG_OLLAMA_URL` (localhost:11434)
- `CV_RAG_CHUNK_MAX_CHARS` (1200), `CV_RAG_CHUNK_OVERLAP` (200)
- Storage paths: `CV_RAG_DATA_DIR`, `CV_RAG_PDF_DIR`, `CV_RAG_TEI_DIR`, `CV_RAG_METADATA_DIR`
- OpenAlex: `OPENALEX_API_KEY` (optional; resolver fails fast on auth-required responses)

## External Services

Qdrant (vector DB), GROBID (PDF parsing), and Ollama (embeddings + LLM) must be running. Use `cv-rag doctor` to verify connectivity.

## Tooling

- **Linter**: ruff (configured in `pyproject.toml`). `B008` ignored for Typer defaults.
- **Type checker**: mypy (configured in `pyproject.toml`, strict mode).
- **Tests**: pytest. All tests are offline (monkeypatched). Run with `uv run pytest -q`.
