# ADR-002: Hybrid retrieval via Reciprocal Rank Fusion

**Date**: 2026-02-15
**Status**: accepted

## Context

Pure vector search misses keyword-exact matches (paper names, model acronyms). Pure keyword search misses semantic similarity. Need a fusion strategy that doesn't require a learned ranker.

## Decision

`HybridRetriever` in `cv_rag/retrieval/hybrid.py` merges Qdrant vector search (cosine) with SQLite FTS5 keyword search (BM25) using Reciprocal Rank Fusion (`rrf_k=60`). Additional scoring layers:
- Tier boosting: Tier 0 +0.25, Tier 1 +0.10 (from `paper_metrics`)
- Section boosting: configurable bonus for method/training/results sections
- Per-document chunk quota via `max_per_doc`
- Deduplication by `(arxiv_id, section_title, chunk_id)`

## Consequences

- Positive: No training required, deterministic, combines dense+sparse strengths
- Negative: Requires both Qdrant and SQLite; latency is sum of both lookups
- Neutral: `rrf_k=60` is a reasonable default; not currently tunable per query

## Affected Files

- `cv_rag/retrieval/hybrid.py` — `HybridRetriever.retrieve()`
- `cv_rag/retrieval/relevance.py` — entity-token extraction, relevance guard
- `cv_rag/storage/qdrant.py` — vector search
- `cv_rag/storage/sqlite.py` — FTS5 keyword search + tier lookup
