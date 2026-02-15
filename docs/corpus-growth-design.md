# Multi-Source Corpus Growth Design

## Scope

This design makes `arxiv`, `doi`, and direct `pdf_url` first-class corpus inputs with one canonical document identity, append-only discovery/resolution artifacts, and idempotent canonical state updates.

Primary implementation packages:

- `cv_rag/seeding/` discovery + normalization + resolution
- `cv_rag/storage/` canonical graph + queue + events
- `cv_rag/ingest/` fulltext parsing/indexing from ready candidates
- `cv_rag/retrieval/` ranking with tier + provenance boosts
- `cv_rag/interfaces/cli/` additive corpus command group
- `cv_rag/interfaces/api/` `doc_id` response compatibility

## Sources

| Source kind | Entry points | Output type |
|---|---|---|
| Curated GitHub repos | `seed-awesome`, `corpus discover-awesome` | arXiv IDs, DOIs |
| VisionBib pages | `seed visionbib`, `corpus discover-visionbib` | DOIs, direct PDF URLs, arXiv links |
| OpenAlex DOI resolution | `resolve-dois`, `corpus resolve-openalex` | arXiv mapping and/or OA PDF URL |
| Existing arXiv ingest paths | `ingest`, `ingest-ids`, `ingest-jsonl` | arXiv metadata + PDF |

## Reference Types And Canonical IDs

Supported reference types:

- `arxiv`
- `doi`
- `pdf_url`

Canonical `doc_id` format:

- `axv:{arxiv_id_with_version}`
- `doi:{normalized_doi}`
- `url:{sha256(normalized_pdf_url)}`

Compatibility rule:

- API and retrieval payloads include `doc_id`.
- `arxiv_id` remains available when known.

## Canonical Data Model

SQLite canonical tables (source of truth):

- `documents` (`doc_id`, identifiers, status, provenance, confidence)
- `document_sources` (many-to-one source lineage)
- `ingest_candidates` (queue state, retries, scheduling)
- `reference_events` (discovery events by run)
- `resolution_events` (resolver outcomes by run)

Indexing tables:

- `papers`, `chunks`, `chunks_fts`, `paper_metrics`

Event artifacts per run:

- `data/curation/runs/{run_id}/awesome_references.jsonl`
- `data/curation/runs/{run_id}/visionbib_references.jsonl`
- `data/curation/runs/{run_id}/openalex_resolution.jsonl`

## Workflow By Reference Type

### arXiv

1. Discover arXiv IDs from feed/repo/page.
2. Normalize to base/versioned form.
3. Canonicalize to `doc_id=axv:{id_with_version}`.
4. Queue as `ready` when versioned, otherwise keep `discovered` until resolved.
5. Ingest: download PDF, parse TEI, chunk, embed, upsert SQLite/Qdrant.
6. Retrieval ranking uses fused score + tier boost + provenance boost.

### DOI

1. Discover DOI from text/link and normalize.
2. Canonicalize to `doc_id=doi:{doi}`.
3. Resolve with OpenAlex.
4. If arXiv mapping exists, promote canonical `axv:` doc and mark DOI alias as resolved.
5. If no arXiv mapping but OA PDF exists, keep `doi:` doc and mark `ready`.
6. If no fulltext, mark `blocked` and schedule retry.

### Direct PDF URL

1. Discover and normalize URL.
2. Canonicalize to `doc_id=url:{sha256(url)}`.
3. If URL maps to arXiv path, normalize into arXiv flow; otherwise remain `url:`.
4. Queue as `ready` with `best_pdf_url`.
5. Ingest and index using `doc_id` even when `arxiv_id` is synthetic/non-arXiv.

### Metadata-only unresolved

1. Persist canonical document + source lineage.
2. Keep candidate in `blocked`/`discovered` backlog.
3. Never index chunks until fulltext is available.
4. Retry on `next_retry_unix` until max retry limit.

## Ranking Policies

### Ingestion Queue Priority

`priority = source_weight + resolution_weight + freshness_bonus + confidence_bonus - retry_penalty`

Configured defaults:

- source: `curated_repo=1.00`, `arxiv_feed=0.90`, `openalex_resolved=0.70`, `scraped_pdf=0.50`
- resolution: `arxiv_versioned=0.40`, `oa_pdf=0.30`, `pdf_only=0.20`, `metadata_only=0.00`
- `freshness_bonus = max(0, 0.2 - age_days/365)`
- `retry_penalty = 0.15 * retry_count`

### Retrieval Boosting

Final score layers:

1. RRF fused retrieval score
2. tier boost (`paper_metrics.tier`)
3. provenance boost (`curated`, `canonical_api`, `scraped`)

Default provenance boosts:

- curated: `+0.08`
- canonical API: `+0.05`
- scraped: `+0.02`

## Extensibility Rules

To add a new source:

1. Implement parser/resolver in `cv_rag/seeding/` emitting `ReferenceRecord`/`ResolvedReference`.
2. Reuse `SQLiteStore.upsert_reference_graph(...)` for canonical writes.
3. Map source kind to provenance classification.
4. Add CLI wiring in `cv_rag/interfaces/cli/commands/corpus.py` and `app.py`.
5. Add test coverage for discovery, resolution, queue behavior, and retrieval impact.

Dependency guardrail:

- Domain packages must not import from `interfaces/` or `app/`.

## Validation Matrix

Required tested behaviors:

- canonical dedupe across repeated discoveries
- DOI->arXiv canonical merge behavior
- blocked retry scheduling and retry-limit failure
- candidate ingest state transitions
- `url:` documents indexed/retrievable with `doc_id`
- provenance boosts affecting retrieval order
