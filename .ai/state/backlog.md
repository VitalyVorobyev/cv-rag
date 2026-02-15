# Task Board

Last updated: 2026-02-15

## In Progress

(none)

## Backlog

### [TASK-010] Add MLX runner tests
- **Priority**: medium
- **Packages**: `answer/mlx_runner.py`
- **Description**: No tests for subprocess-based generation. Test `mlx_generate` and `mlx_generate_stream` with monkeypatched `subprocess.run`/`Popen`. Test error paths (missing runtime, empty output).

### [TASK-011] Add AnswerService integration tests
- **Priority**: medium
- **Packages**: `answer/service.py`
- **Description**: No dedicated tests for `AnswerService.run()` or `.stream()`. Test the full flow: prepare → route → retrieve → generate → validate. Monkeypatch retriever and MLX runner.

### [TASK-012] Add IngestService integration tests
- **Priority**: low
- **Packages**: `ingest/service.py`
- **Description**: No dedicated tests for `IngestService`. Test `ingest_latest`, `ingest_ids`, `ingest_jsonl` with monkeypatched arXiv client, GROBID, and stores.

## Done

### [TASK-001] Initial ingestion pipeline
- **Commits**: c2703c1, 5870e93
- **Description**: First commit + arXiv/Ollama/Qdrant compatibility fixes

### [TASK-002] Ingest-ids, relevance guard, answer refusal
- **Commit**: 553fb9b
- **Description**: Explicit paper ingestion by ID, retrieval relevance filtering, answer refusal path

### [TASK-003] Retrieval scoring and answer grounding
- **Commits**: a0a4e27, 70c8ed4
- **Description**: Improved retrieval scoring, RRF merging, citation enforcement

### [TASK-004] Skip-ingested ingest and eval harness
- **Commit**: 68060bd
- **Description**: Dedup on ingest, evaluation framework

### [TASK-005] Tiered corpus curator
- **Commit**: e3c0e79
- **Description**: Semantic Scholar integration, citation-based tiering

### [TASK-006a] Awesome-list seeder
- **Commit**: ff778e0
- **Description**: GitHub awesome-list paper extraction and seeding

### [TASK-007a] Answer routing with mode-specific prompts
- **Commits**: 386fabc, a9fec9f
- **Description**: 5 answer modes (single/compare/survey/implement/evidence), rules+LLM+hybrid routing

### [TASK-008a] Web UI
- **Commit**: fabfbe2
- **Description**: FastAPI backend + React frontend with SSE streaming

### [TASK-009a] DOI/OpenAlex seeding and JSONL ingest
- **Commit**: d165486
- **Description**: OpenAlex DOI resolution, JSONL ingest support

### [TASK-010a] Exact-version dedupe
- **Commit**: 134dee0
- **Description**: Treat versioned arXiv IDs as distinct, legacy migration

### [TASK-011a] Domain packages refactor
- **Commit**: d6d1850
- **Description**: Reorganize into domain-driven packages with thin CLI/API interfaces

### [TASK-006] QdrantStore unit tests
- **Completed**: 2026-02-15
- **Test file**: `tests/test_qdrant_store.py` (16 tests)
- **Description**: Point ID normalization, ensure_collection, upsert batching, search with legacy fallback

### [TASK-007] API router tests
- **Completed**: 2026-02-15
- **Test file**: `tests/test_api_routers.py` (11 tests)
- **Description**: Health, search, papers, stats routers + _parse_authors utility

### [TASK-008] GROBID client and TEI extract tests
- **Completed**: 2026-02-15
- **Test file**: `tests/test_grobid_tei.py` (12 tests)
- **Description**: TEI XML section extraction, whitespace normalization, GROBID HTTP retry

### [TASK-009] OllamaEmbedClient tests
- **Completed**: 2026-02-15
- **Test file**: `tests/test_embeddings.py` (19 tests)
- **Description**: Batch/legacy endpoints, model fallback, error extraction, batch splitting
