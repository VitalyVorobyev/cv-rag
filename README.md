# cv-rag — Local CV literature assistant

A local-first RAG system for computer-vision papers. Ingest arXiv PDFs, parse them with GROBID, embed and index with Qdrant + SQLite, and answer questions using a local MLX LLM — all on a MacBook.

Answers are **grounded in the literature** with inline `[S#]` citations to retrieved source chunks.

## Quickstart

### Prerequisites

| Dependency | Purpose |
|---|---|
| [uv](https://docs.astral.sh/uv/) | Python package manager |
| [Docker](https://docs.docker.com/get-docker/) (or [Colima](https://github.com/abiosoft/colima)) | Run Qdrant vector database |
| [GROBID](https://grobid.readthedocs.io/) | PDF → structured text |
| [Ollama](https://ollama.com/) | Local embeddings |
| An MLX-compatible model | Local LLM generation (e.g. `mlx-community/Qwen2.5-7B-Instruct-4bit`) |

### 1. Install

```bash
uv sync
```

For the web UI, include the `web` extra:

```bash
uv sync --extra web
```

### 2. Start services

```bash
# Qdrant
docker compose up -d

# GROBID (in your grobid checkout)
./gradlew run

# Ollama
ollama serve
ollama pull nomic-embed-text   # or another embedding model
```

### 3. Verify connectivity

```bash
uv run cv-rag doctor
```

### 4. Ingest papers

```bash
# Latest cs.CV papers from arXiv
uv run cv-rag ingest -n 20

# Specific papers by arXiv ID
uv run cv-rag ingest-ids 2104.00680 1911.11763

# From a JSONL seed file
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl
```

All ingest commands skip already-ingested versions by default. Use `--no-skip-ingested` to re-ingest.

### 5. Ask questions

```bash
# Inspect retrieval
uv run cv-rag query "vision transformers" --top-k 10

# Answer with citations
uv run cv-rag answer \
  "Summarize the key idea of LoFTR and compare with SuperGlue." \
  --mode auto \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --show-sources
```

Answer modes: `auto`, `single`, `compare`, `survey`, `implement`, `evidence`.
In `auto` mode, a cheap retrieval pass routes the question to the best mode.

## Web UI

A browser-based interface for chat Q&A, corpus browsing, and system health monitoring.

```bash
uv sync --extra web
uv run cv-rag serve          # http://127.0.0.1:8000
```

Pages: **Chat** (`/`), **Papers** (`/papers`), **Stats** (`/stats`), **Health** (`/health`).

For frontend development with hot reload:

```bash
# Terminal 1: backend with auto-reload
uv run cv-rag serve --reload

# Terminal 2: Vite dev server (proxies /api → :8000)
cd web && npm install && npm run dev   # http://localhost:5173
```

Production build: `cd web && npm run build` (outputs to `cv_rag/interfaces/api/static/`).

## Architecture

### Ingestion pipeline

arXiv API → PDF download → GROBID (TEI XML) → section extraction → chunking → Ollama embeddings → Qdrant + SQLite

### Hybrid retrieval

`HybridRetriever` merges Qdrant vector search (cosine) with SQLite FTS5 keyword search (BM25) via Reciprocal Rank Fusion (RRF). Per-document chunk quotas, section boosting, and deduplication keep results diverse and relevant.

### Answer generation

1. Cheap prelim retrieval + answer routing (`single|compare|survey|implement|evidence`)
2. Mode-specific prompt with `[S1..Sk]` source excerpts and citation rules
3. MLX LLM generation
4. Citation validation (with repair loop if needed)

### Code layout

```
cv_rag/
  shared/          # settings, errors, HTTP retry helpers
  storage/         # SQLite + Qdrant adapters
  ingest/          # arXiv client, PDF/TEI/chunk pipeline, dedupe
  retrieval/       # hybrid retrieval + relevance filtering
  answer/          # routing, prompts, citations, MLX runner
  curation/        # Semantic Scholar enrichment and tiering
  seeding/         # awesome-list + OpenAlex DOI seeding
  embeddings.py    # Ollama embedding client
  interfaces/
    cli/           # Typer CLI entrypoint + command handlers
    api/           # FastAPI app, routers, schemas, static SPA
web/               # React + TypeScript + Tailwind frontend (Vite)
```

## Corpus seeding

Beyond basic `ingest`, cv-rag supports bulk corpus building from curated sources.

### Awesome-list seeding

Extract arXiv IDs and DOIs from GitHub awesome lists:

```bash
uv run cv-rag seed-awesome \
  --sources data/curation/awesome_sources.txt \
  --out-dir data/curation

# Then ingest
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl
```

### DOI resolution via OpenAlex

Resolve DOI seeds to Open Access PDF URLs:

```bash
export OPENALEX_API_KEY=...   # optional but recommended
uv run cv-rag resolve-dois \
  --dois data/curation/tierA_dois.txt \
  --out-dir data/curation
```

### VisionBib seeding

Scrape bibliography pages from VisionBib:

```bash
uv run cv-rag seed visionbib \
  --sources data/curation/visionbib_sources.txt \
  --out-dir data/curation/visionbib
```

### Corpus curation (Semantic Scholar)

Enrich papers with citation counts and venue metadata:

```bash
uv run cv-rag curate --refresh-days 30
```

## Configuration

All settings are in `cv_rag/shared/settings.py` via `CV_RAG_*` environment variables.

| Variable | Default | Purpose |
|---|---|---|
| `CV_RAG_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `CV_RAG_GROBID_URL` | `http://localhost:8070` | GROBID endpoint |
| `CV_RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `CV_RAG_DATA_DIR` | `./data` | Root data directory |
| `CV_RAG_CHUNK_MAX_CHARS` | `1200` | Max characters per chunk |
| `CV_RAG_CHUNK_OVERLAP` | `200` | Chunk overlap in characters |
| `OPENALEX_API_KEY` | — | OpenAlex API key (optional) |

## Development

```bash
# Run tests (all offline, monkeypatched)
uv run pytest

# Lint
uv run ruff check cv_rag/ tests/
uv run ruff check --fix cv_rag/ tests/   # auto-fix

# Verbose logging for any command
uv run cv-rag -v ingest --limit 5
```

## Troubleshooting

**GROBID connection refused** — Ensure the service is running on port 8070: `curl http://localhost:8070/api/isalive`

**Retrieval returns unrelated papers** — Lower `--k`, increase `--max-per-doc`, or raise `--section-boost` to prioritize Methods/Loss sections.

**"Answer has no inline citations"** — The answer failed citation validation. Try adjusting `--temperature` or `--top-p`, or check that retrieved sources are relevant with `cv-rag query`.

## License

TBD
