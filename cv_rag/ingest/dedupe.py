from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from cv_rag.ingest.arxiv_client import extract_version, fetch_papers_by_ids, normalize_arxiv_id
from cv_rag.ingest.models import PaperMetadata
from cv_rag.shared.settings import Settings
from cv_rag.storage.sqlite import SQLiteStore


@dataclass(slots=True)
class LegacyVersionMigrationStats:
    legacy_rows_found: int = 0
    migrated: int = 0
    unresolved: int = 0


@dataclass(slots=True)
class ExplicitIngestStats:
    requested_ids: int
    metadata_returned: int
    resolved_to_version: int
    unresolved: int
    skipped_existing: int = 0
    to_ingest: int = 0


def canonical_requested_ids(raw_ids: list[str]) -> list[str]:
    requested_ids: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        base_id = normalize_arxiv_id(raw_id)
        if not base_id:
            continue
        version = extract_version(raw_id)
        canonical = f"{base_id}{version or ''}"
        if canonical in seen:
            continue
        seen.add(canonical)
        requested_ids.append(canonical)
    return requested_ids


def load_ingested_versions(settings: Settings) -> set[str]:
    sqlite_store = SQLiteStore(settings.sqlite_path)
    try:
        sqlite_store.create_schema()
        return sqlite_store.get_ingested_versioned_ids()
    finally:
        sqlite_store.close()


def find_and_migrate_legacy_versions(settings: Settings) -> LegacyVersionMigrationStats:
    sqlite_store = SQLiteStore(settings.sqlite_path)
    try:
        sqlite_store.create_schema()
        legacy_base_ids = sqlite_store.list_legacy_unversioned_arxiv_ids()
    finally:
        sqlite_store.close()

    stats = LegacyVersionMigrationStats(legacy_rows_found=len(legacy_base_ids))
    if not legacy_base_ids:
        return stats

    resolved_papers = fetch_papers_by_ids(
        ids=legacy_base_ids,
        arxiv_api_url=settings.arxiv_api_url,
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.user_agent,
        max_retries=settings.arxiv_max_retries,
        backoff_start_seconds=settings.arxiv_backoff_start_seconds,
        backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
        resolve_unversioned_to_latest=True,
    )
    by_base = {paper.arxiv_id: paper for paper in resolved_papers if paper.arxiv_id}

    sqlite_store = SQLiteStore(settings.sqlite_path)
    try:
        sqlite_store.create_schema()
        for base_id in legacy_base_ids:
            paper = by_base.get(base_id)
            if paper is None:
                stats.unresolved += 1
                continue

            resolved_version = extract_version(paper.arxiv_id_with_version)
            if resolved_version is None:
                stats.unresolved += 1
                continue

            updated = sqlite_store.update_paper_version_fields(
                arxiv_id=base_id,
                arxiv_id_with_version=paper.arxiv_id_with_version,
                version=paper.version or resolved_version,
            )
            if updated:
                stats.migrated += 1
            else:
                stats.unresolved += 1
    finally:
        sqlite_store.close()

    return stats


def filter_papers_by_exact_version(
    papers: list[PaperMetadata],
    ingested_versions: set[str],
) -> tuple[list[PaperMetadata], int]:
    if not papers or not ingested_versions:
        return papers, 0

    selected: list[PaperMetadata] = []
    skipped = 0
    for paper in papers:
        versioned = paper.arxiv_id_with_version.strip()
        if versioned and versioned in ingested_versions:
            skipped += 1
            continue
        selected.append(paper)
    return selected, skipped


def build_explicit_ingest_stats(
    requested_ids: list[str],
    papers: list[PaperMetadata],
) -> ExplicitIngestStats:
    resolved_to_version = 0
    unresolved = 0

    for requested_id, paper in zip(requested_ids, papers, strict=False):
        if extract_version(requested_id):
            continue
        if extract_version(paper.arxiv_id_with_version):
            resolved_to_version += 1
        else:
            unresolved += 1

    if len(papers) < len(requested_ids):
        for requested_id in requested_ids[len(papers) :]:
            if extract_version(requested_id) is None:
                unresolved += 1

    return ExplicitIngestStats(
        requested_ids=len(requested_ids),
        metadata_returned=len(papers),
        resolved_to_version=resolved_to_version,
        unresolved=unresolved,
    )


def load_arxiv_ids_from_jsonl(
    jsonl_path: Path,
    *,
    preferred_fields: tuple[str, ...] = ("arxiv_id", "base_id"),
) -> tuple[list[str], int]:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    ids: list[str] = []
    seen: set[str] = set()
    skipped_records = 0

    for line_no, raw_line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{jsonl_path}:{line_no}: invalid JSON ({exc})") from exc

        value: str | None = None
        if isinstance(payload, dict):
            for field in preferred_fields:
                candidate = payload.get(field)
                if isinstance(candidate, str) and candidate.strip():
                    value = candidate.strip()
                    break
        elif isinstance(payload, str) and payload.strip():
            value = payload.strip()

        if not value:
            skipped_records += 1
            continue

        if value in seen:
            continue
        seen.add(value)
        ids.append(value)

    return ids, skipped_records
