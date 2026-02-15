# Corpus Update Policy

## Goals

- Preserve raw discovery/resolution provenance.
- Keep canonical state idempotent and deterministic.
- Avoid destructive rewrites during normal runs.
- Prevent unnecessary re-parse/re-embed work.

## Policy 1: Append-Only Run Artifacts

Every discovery/resolution run writes immutable files under:

- `data/curation/runs/{run_id}/`

Examples:

- `awesome_references.jsonl`
- `visionbib_references.jsonl`
- `openalex_resolution.jsonl`

Rules:

- Never rewrite prior run directories.
- New executions must use a new `run_id`.
- Re-running with a distinct `run_id` is additive.

## Policy 2: Canonical State Uses Idempotent Upserts

Canonical tables (`documents`, `document_sources`, `ingest_candidates`, event tables) are updated by deterministic keys:

- `documents`: `doc_id`
- `document_sources`: `(doc_id, source_kind, source_ref)`
- `ingest_candidates`: `doc_id`

Rules:

- Re-processing identical references must not create duplicate canonical docs.
- Resolution updates can improve status/confidence but should not remove history.
- DOI aliases that resolve to canonical arXiv docs are marked resolved (not queued for duplicate ingest).

## Policy 3: Derived Snapshots Are Replaceable Outputs

Compatibility snapshots (for legacy commands/workflows), e.g.:

- `tierA_*.txt`
- seed snapshot files in output directories

Rules:

- These are derived from canonical/event outputs.
- They may be regenerated atomically and replaced.
- They are not source-of-truth logs.

## Policy 4: Fulltext/Index Write Minimization

Rules:

- Do not re-embed/re-parse unchanged content.
- Reprocessing is allowed when forced (`--force-*`) or when content hash changes.
- Missing fulltext remains backlog (`blocked`), not silently dropped.

## Policy 5: Retry And Failure Lifecycle

Candidate lifecycle:

- `discovered` -> `ready` -> `ingested`
- `ready` -> `blocked` (transient failure or missing fulltext)
- `blocked` -> retry by `next_retry_unix`
- after retry limit -> `failed`

Rules:

- `retry_count` increments on blocked/failed transitions.
- `next_retry_unix` uses configured retry days.
- `last_error` stores transition reason for auditability.

## Maintenance Boundaries

Allowed in normal pipeline runs:

- append events
- idempotent upserts
- queue status transitions

Not allowed in normal runs:

- destructive truncation of canonical tables
- deleting historical run artifacts
- schema-reset operations

Maintenance/reset operations must be explicit operator actions (for example, full local reindex).
