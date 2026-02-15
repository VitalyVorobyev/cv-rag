# Handoff: Implementer -> Reviewer

**Task**: TASK-012 — Add IngestService integration tests  
**Date**: 2026-02-15

## Files Changed

- `tests/test_ingest_service.py` — created — offline integration-style tests for `IngestService.ingest_latest`, `.ingest_ids`, and `.ingest_jsonl`

## Summary

Implemented deterministic service-level tests for `cv_rag/ingest/service.py` with injected fakes (no network, no DB):

- `ingest_latest(skip_ingested=False)`:
  - verifies migration/version-loading/filter hooks are not called
  - verifies `skip_arxiv_id_with_version=None` is passed to fetch
  - verifies `migration=None` and `skipped_after_fetch=0`
- `ingest_latest(skip_ingested=True)`:
  - verifies migration + version-loading are called
  - verifies ingested-version set is passed to fetch and exact-version filter
  - verifies filtered papers and `skipped_after_fetch`
- `ingest_ids()`:
  - empty canonical IDs returns empty selection + zeroed stats without fetch/migration
  - `skip_ingested=False`: fetch uses `resolve_unversioned_to_latest=False`, no exact-version filter, `to_ingest=len(papers)`
  - `skip_ingested=True`: migration/version-loading/filter hooks run, fetch uses `resolve_unversioned_to_latest=True`, stats `skipped_existing` + `to_ingest` are populated
- `ingest_jsonl()`:
  - verifies JSONL `limit` is applied before canonicalization
  - verifies `skipped_records` is preserved in result
  - verifies behavior mirrors `ingest_ids()` for both skip toggles
  - verifies `FileNotFoundError` and `ValueError` from JSONL loader are propagated unchanged

## Deviations / Open Questions

- No deviations from spec.
- No open questions.

## Verification

- `uv run ruff check cv_rag/`: passed
- `uv run ruff check tests/test_ingest_service.py`: passed
- `uv run pytest -q`: passed (`167 passed`)
