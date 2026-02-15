# ADR-007: Multi-Source Corpus Growth Canonical Model

**Date**: 2026-02-15
**Status**: accepted

## Context

The corpus pipeline had strong arXiv ingest support but weak first-class handling for DOI and direct PDF URL references. Seed artifacts were often rewritten, provenance was fragmented, and DOI/PDF discoveries were not consistently connected to candidate ingestion state.

We need a design that:

- supports `arxiv`, `doi`, and `pdf_url` as first-class references
- preserves raw discovery/resolution history
- keeps canonical state deterministic and idempotent
- allows additive CLI/API evolution without breaking existing workflows

## Decision

Adopt a canonical multi-source document graph with deterministic `doc_id` values and append-plus-upsert update policy.

Canonical identity:

- `axv:{arxiv_id_with_version}`
- `doi:{normalized_doi}`
- `url:{sha256(normalized_pdf_url)}`

Persistence model:

- append run-scoped event artifacts under `data/curation/runs/{run_id}/`
- upsert canonical state in SQLite (`documents`, `document_sources`, `ingest_candidates`, events)
- preserve compatibility outputs (`tierA_*.txt`) as derived snapshots

Queue/ranking model:

- deterministic candidate priority function from source, resolution confidence/type, age, retries
- retrieval score layering with existing tier boost plus new provenance boost

Compatibility policy:

- keep existing seed/resolve/ingest CLI commands operational
- add additive `corpus` command group
- include `doc_id` in API/search/answer payloads while preserving `arxiv_id` when available

## Consequences

- Positive: Unified workflow across reference types with explicit lifecycle and provenance.
- Positive: Safer updates (append-only history + idempotent canonical writes).
- Positive: Better ranking control via provenance-aware boosts.
- Negative: Increased schema and queue logic complexity.
- Negative: Local migration/reindex cost for existing stores.

## Affected Files

- `cv_rag/storage/sqlite.py` — canonical graph tables, queue lifecycle, event upserts
- `cv_rag/storage/repositories.py` — canonical dataclasses, doc_id builders, priority scoring
- `cv_rag/seeding/awesome.py` — discovery records + run artifacts + graph upsert
- `cv_rag/seeding/visionbib.py` — discovery/resolution records + graph upsert
- `cv_rag/seeding/openalex.py` — DOI resolution records + graph upsert
- `cv_rag/ingest/service.py` — queue-driven ingest entrypoints and candidate transitions
- `cv_rag/ingest/pdf_pipeline.py` — doc_id-aware indexing for chunks/vector payloads
- `cv_rag/retrieval/hybrid.py` — provenance boosts in retrieval scoring
- `cv_rag/interfaces/cli/commands/corpus.py` — additive multi-source corpus commands
- `cv_rag/interfaces/api/schemas.py` — `doc_id` API compatibility field
