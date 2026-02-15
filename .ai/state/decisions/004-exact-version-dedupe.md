# ADR-004: Exact-version deduplication

**Date**: 2026-02-15
**Status**: accepted

## Context

ArXiv papers have multiple versions (v1, v2, ...). Early ingestion stored only base IDs, causing re-ingestion of already-processed papers. Need a dedup strategy that handles versioning correctly.

## Decision

Treat `arxiv_id_with_version` (e.g., `2104.00680v2`) as the dedup key. Different versions are considered distinct papers. `--skip-ingested` (default) checks against the versioned ID set. `--no-skip-ingested` forces re-ingestion.

Legacy migration (`find_and_migrate_legacy_versions`) resolves old unversioned records to their latest version by querying the arXiv API.

For unversioned explicit IDs, version resolution is attempted first; if it fails, ingestion continues with a warning.

## Consequences

- Positive: No duplicate processing, clean version tracking, backward-compatible migration
- Negative: Storage increases if multiple versions ingested; version resolution adds API calls
- Neutral: Default behavior (skip-ingested) is safe; override available

## Affected Files

- `cv_rag/ingest/dedupe.py` — `filter_papers_by_exact_version()`, `find_and_migrate_legacy_versions()`
- `cv_rag/ingest/service.py` — `IngestService.ingest_ids()`, `ingest_jsonl()`, `ingest_latest()`
- `cv_rag/storage/sqlite.py` — `get_ingested_versioned_ids()`, `list_legacy_unversioned_arxiv_ids()`
