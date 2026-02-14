# cv-rag — Local CV literature assistant (RAG over arXiv)

A local-first Retrieval-Augmented Generation (RAG) system for computer-vision papers:

* Sync and ingest arXiv papers (PDF)
* Parse papers into structured text with **GROBID**
* Chunk + embed passages (local embeddings via **Ollama**)
* Store vectors in **Qdrant**
* Store metadata + keyword index in **SQLite**
* Answer questions using a local **MLX** LLM with **inline citations** to retrieved sources

> Goal: answer deep CV questions **grounded in the literature**, with citations, using a MacBook-class machine.

---

## Architecture

**Ingest**

1. arXiv → download PDFs
2. PDF → TEI XML via GROBID
3. TEI → sectioned text
4. sectioned text → chunks
5. chunks → embeddings (Ollama)
6. embeddings → Qdrant points (+ payload)
7. metadata/FTS → SQLite

**Query / Answer**

1. hybrid retrieve: SQLite keyword + Qdrant vector
2. cheap prelim retrieval + answer routing (`single|compare|survey|implement|evidence`)
3. build mode-specific prompt with `[S1..Sk]` excerpts + citation rules
4. generate answer with MLX LLM
5. validate citations (and repair if needed)

---

## Requirements

* macOS (Apple Silicon recommended)
* Python + **uv**
* Docker (via Colima or Docker Desktop) for Qdrant
* Java (OpenJDK) for GROBID
* Ollama for embeddings
* MLX-LM for local generation

---

## Quickstart

### 1) Create environment

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

### 2) Start Qdrant (Docker)

```bash
colima start --cpu 6 --memory 12 --disk 80
docker compose up qdrant
```

### 3) Start GROBID (recommended on Apple Silicon)

Run GROBID as a local service (Gradle):

```bash
# in your grobid checkout
./gradlew run
```

Verify:

```bash
curl -s http://localhost:8070/api/isalive
curl -s http://localhost:8070/api/version
```

### 4) Start Ollama + pull an embedding model

```bash
ollama serve   # if not already running
ollama pull embeddinggemma
```

### 5) Health check

```bash
uv run cv-rag doctor
```

---

## Usage

### Ingest latest papers

```bash
uv run cv-rag ingest -n 50
```

`-n` is a target number of **new** papers after filtering out already ingested versions.
By default, `ingest` skips versions already present in SQLite. Disable this with:

```bash
uv run cv-rag ingest -n 50 --no-skip-ingested
```

### Ingest specific papers by arXiv id

```bash
uv run cv-rag ingest-ids 2104.00680 1911.11763
```

By default, `ingest-ids` skips exact versions already present in SQLite (`arxiv_id_with_version`).
For unversioned inputs, cv-rag resolves to the latest arXiv version when possible before dedupe.
If version resolution fails, ingest continues in best-effort mode and prints a warning.

```bash
uv run cv-rag ingest-ids 2104.00680 1911.11763 --no-skip-ingested
```

### Ingest papers from JSONL seed files

```bash
# Works with records that contain arxiv_id or base_id
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl

# Optional cap for quick batches
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl --limit 200

# Disable exact-version dedupe for this run
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl --no-skip-ingested
```

`ingest-jsonl` uses the same exact-version skip behavior and best-effort resolution warning semantics as `ingest-ids`.

### Inspect retrieval

```bash
uv run cv-rag query "SuperGlue loss negative log-likelihood dustbin Sinkhorn" --top-k 10
```

### Answer with citations (local MLX model)

```bash
uv run cv-rag answer \
  "Summarize the key idea and training objective of LoFTR. Compare with SuperGlue." \
  --mode auto \
  --router-strategy hybrid \
  --router-top-k 12 \
  --k 12 \
  --max-per-doc 4 \
  --section-boost 0.1 \
  --temperature 0.1 \
  --top-p 0.9 \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --show-sources
```

`answer` supports explicit mode override:

```bash
uv run cv-rag answer "Compare LoFTR and SuperGlue" --mode compare --model mlx-community/Qwen2.5-7B-Instruct-4bit
```

Modes: `auto`, `single`, `compare`, `survey`, `implement`, `evidence`.
In `auto`, routing runs after a cheap retrieval pass and may downgrade compare requests if cross-paper evidence is too sparse.

### Seed from awesome lists + resolve OA PDFs

```bash
# Extract arXiv IDs + DOIs from GitHub awesome lists
uv run cv-rag seed-awesome \
  --sources data/curation/awesome_sources.txt \
  --out-dir data/curation

# Equivalent nested command
uv run cv-rag seed awesome \
  --sources data/curation/awesome_sources.txt \
  --out-dir data/curation

# Resolve DOI seeds to Open Access PDF URLs via OpenAlex
export OPENALEX_API_KEY=...   # optional but recommended
uv run cv-rag resolve-dois \
  --dois data/curation/tierA_dois.txt \
  --out-dir data/curation
```

Seeding outputs:

* `data/curation/awesome_seed.jsonl` (arXiv provenance)
* `data/curation/awesome_seed_doi.jsonl` (DOI provenance)
* `data/curation/tierA_seed.txt` (legacy arXiv IDs, kept for compatibility)
* `data/curation/tierA_arxiv.txt` (preferred arXiv IDs)
* `data/curation/tierA_dois.txt` (DOI seeds)

You can ingest the arXiv seed JSONL directly with:

```bash
uv run cv-rag ingest-jsonl --source data/curation/awesome_seed.jsonl
```

OpenAlex resolution outputs:

* `data/curation/openalex_resolved.jsonl`
* `data/curation/tierA_urls_openalex.txt`
* `data/curation/openalex_cache/` (response cache)

### Curate corpus tiers (Semantic Scholar metadata)

```bash
uv run cv-rag curate \
  --refresh-days 30 \
  --tier0-venues data/venues_tier0.txt
```

### Run evaluation suite

```bash
uv run cv-rag eval --suite eval/questions.yaml
```

This runs retrieval + answer generation for each case, validates inline citations, and exits non-zero if any case fails.

### Show local database statistics

```bash
uv run cv-rag stats
```

Use `--top-venues 0` to skip venue frequency output.

---

## Configuration

Config lives in `cv_rag/config.py`. Typical knobs:

* Service URLs:

  * `QDRANT_URL` (default `http://localhost:6333`)
  * `GROBID_URL` (default `http://localhost:8070`)
  * `OLLAMA_URL` (default `http://localhost:11434`)
* Paths under `./data/`:

  * `data/pdfs/`, `data/tei/`, `data/metadata/`
* Retrieval:

  * `--k`, `--max-per-doc`, `--section-boost`
  * `answer` routing: `--mode`, `--router-model`, `--router-strategy`, `--router-top-k`
* OpenAlex:
  * `OPENALEX_API_KEY` (optional; command fails fast on auth-required responses)

---

## Data layout

```
data/
  pdfs/       # downloaded arXiv PDFs
  tei/        # GROBID TEI XML outputs
  metadata/   # arXiv metadata snapshots
  curation/
    awesome_seed.jsonl
    awesome_seed_doi.jsonl
    tierA_seed.txt         # legacy arXiv seed file
    tierA_arxiv.txt        # preferred arXiv seed file
    tierA_dois.txt         # DOI seed file
    openalex_resolved.jsonl
    tierA_urls_openalex.txt
    openalex_cache/
  venues_tier0.txt  # whitelist for top-tier venue bonus/tiering
qdrant_storage/  # Qdrant persistent volume
```

---

## Troubleshooting

### GROBID: connection refused

* Ensure the service is running and listening on `8070`.
* Check:

  ```bash
  curl -v http://localhost:8070/api/isalive
  ```

### Retrieval returns unrelated papers

* Lower `--k` (don’t force filling with weak matches)
* Increase `--max-per-doc` so the relevant papers dominate
* Increase `--section-boost` to prioritize `Methods/Loss/Supervision` sections

### “Answer has no inline citations”

* This is considered a failure: tune prompt/citation validator settings so answers must cite `[S#]` per paragraph/claim.

---

## Web UI

A browser-based interface for chat Q&A, corpus browsing, and system monitoring.

### Setup

```bash
# Install web dependencies (FastAPI + Uvicorn)
uv sync --extra web

# Start the server
uv run cv-rag serve          # http://127.0.0.1:8000
```

### Pages

* **Chat** (`/`) — conversational Q&A with streaming answers, clickable `[S#]` citations, and a source panel
* **Papers** (`/papers`) — searchable paginated corpus browser with tier badges
* **Stats** (`/stats`) — paper/chunk counts, tier distribution, top venues
* **Health** (`/health`) — Qdrant, GROBID, Ollama connectivity status

### Frontend development

```bash
# Terminal 1: backend
uv run cv-rag serve --reload

# Terminal 2: frontend (hot-reload, proxies /api → :8000)
cd web && npm install && npm run dev   # http://localhost:5173
```

Production build: `cd web && npm run build` (outputs to `cv_rag/api/static/`, served by FastAPI).

---

## Roadmap

* Better section-aware retrieval and per-paper quotas
* Optional reranking (late interaction / cross-encoder)
* Continuous updates (daily arXiv sync job)
* Export: "answer with bibliography" and "paper-to-notes" flows

---

## License

TBD (add a LICENSE file).
