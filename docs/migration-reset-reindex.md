# Migration: Reset + Reindex (Pre-deployment)

## Scope

This project is pre-deployment, so migration is implemented as a deterministic local reset and rebuild, not online schema migration.

## Command

```bash
uv run cv-rag migrate reset-reindex --yes \
  --backup-dir data/backups \
  --awesome-sources data/curation/awesome_sources.txt \
  --visionbib-sources data/curation/visionbib_sources.txt \
  --dois data/curation/tierA_dois.txt
```

Optional flags:

- `--skip-curate` to skip Semantic Scholar enrichment.
- `--force-grobid` for full TEI reparsing during queue ingest.
- `--embed-batch-size <n>` to override embedding batch size.
- `--cache-only` to restore references from existing run artifacts and ingest only locally cached PDFs (no remote discovery/OpenAlex/PDF fetch).

## Behavior

1. Preflight: validates required input files and service reachability.
2. Optional SQLite backup (`--backup-dir`).
3. Drops existing Qdrant collection (if present).
4. Removes SQLite DB and sidecar files, recreates schema.
5. Rebuilds corpus in fixed order:
   - `corpus discover-awesome`
   - `corpus discover-visionbib`
   - `corpus resolve-openalex`
   - `corpus ingest` loop until no ready candidates
   - `curate` (unless `--skip-curate`)

In `--cache-only` mode:

1. Discovery and OpenAlex resolution are skipped.
2. Reference graph is restored from latest cached run artifacts in `data/curation/runs/` (or explicit `--cache-*-*` paths).
3. Ingest runs in cache-only mode and blocks candidates whose PDFs are not already cached locally.
4. Curation is skipped automatically.

## Report Artifact

Each run writes:

- `data/migrations/<timestamp>-reset-reindex-report.json`

Report fields include step status/duration, backup path, collection info, and aggregate ingest-loop stats.

## Failure and Resume

Runs are resumable by rerunning the command:

- The command always rewrites canonical local stores from scratch.
- On failure, a report with failed step details is still produced.
- Rerun after fixing the failing dependency/input.
