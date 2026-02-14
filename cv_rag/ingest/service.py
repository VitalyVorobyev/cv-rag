from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cv_rag.ingest.arxiv_client import fetch_cs_cv_papers, fetch_papers_by_ids
from cv_rag.ingest.dedupe import (
    ExplicitIngestStats,
    LegacyVersionMigrationStats,
    build_explicit_ingest_stats,
    canonical_requested_ids,
    filter_papers_by_exact_version,
    find_and_migrate_legacy_versions,
    load_arxiv_ids_from_jsonl,
    load_ingested_versions,
)
from cv_rag.ingest.models import PaperMetadata
from cv_rag.shared.settings import Settings


@dataclass(slots=True)
class LatestIngestSelection:
    papers: list[PaperMetadata]
    fetch_stats: dict[str, int]
    migration: LegacyVersionMigrationStats | None
    skipped_after_fetch: int


@dataclass(slots=True)
class ExplicitIngestSelection:
    papers: list[PaperMetadata]
    requested_ids: list[str]
    skipped_records: int
    stats: ExplicitIngestStats
    migration: LegacyVersionMigrationStats | None


class IngestService:
    def __init__(
        self,
        settings: Settings,
        *,
        fetch_latest_fn: Callable[..., list[PaperMetadata]] = fetch_cs_cv_papers,
        fetch_by_ids_fn: Callable[..., list[PaperMetadata]] = fetch_papers_by_ids,
        find_and_migrate_legacy_versions_fn: Callable[
            [Settings], LegacyVersionMigrationStats
        ] = find_and_migrate_legacy_versions,
        load_ingested_versions_fn: Callable[[Settings], set[str]] = load_ingested_versions,
        canonical_requested_ids_fn: Callable[[list[str]], list[str]] = canonical_requested_ids,
        filter_papers_by_exact_version_fn: Callable[
            [list[PaperMetadata], set[str]],
            tuple[list[PaperMetadata], int],
        ] = filter_papers_by_exact_version,
        build_explicit_ingest_stats_fn: Callable[
            [list[str], list[PaperMetadata]],
            ExplicitIngestStats,
        ] = build_explicit_ingest_stats,
        load_arxiv_ids_from_jsonl_fn: Callable[[Path], tuple[list[str], int]] = load_arxiv_ids_from_jsonl,
    ) -> None:
        self.settings = settings
        self._fetch_latest = fetch_latest_fn
        self._fetch_by_ids = fetch_by_ids_fn
        self._find_and_migrate_legacy_versions = find_and_migrate_legacy_versions_fn
        self._load_ingested_versions = load_ingested_versions_fn
        self._canonical_requested_ids = canonical_requested_ids_fn
        self._filter_papers_by_exact_version = filter_papers_by_exact_version_fn
        self._build_explicit_ingest_stats = build_explicit_ingest_stats_fn
        self._load_arxiv_ids_from_jsonl = load_arxiv_ids_from_jsonl_fn

    def ingest_latest(self, *, limit: int, skip_ingested: bool) -> LatestIngestSelection:
        ingested_versions: set[str] = set()
        migration: LegacyVersionMigrationStats | None = None
        if skip_ingested:
            migration = self._find_and_migrate_legacy_versions(self.settings)
            ingested_versions = self._load_ingested_versions(self.settings)

        fetch_stats: dict[str, int] = {}
        papers = self._fetch_latest(
            limit=limit,
            arxiv_api_url=self.settings.arxiv_api_url,
            timeout_seconds=self.settings.http_timeout_seconds,
            user_agent=self.settings.user_agent,
            max_retries=self.settings.arxiv_max_retries,
            backoff_start_seconds=self.settings.arxiv_backoff_start_seconds,
            backoff_cap_seconds=self.settings.arxiv_backoff_cap_seconds,
            skip_arxiv_id_with_version=ingested_versions if skip_ingested else None,
            stats=fetch_stats,
        )

        skipped_after_fetch = 0
        if skip_ingested:
            papers, skipped_after_fetch = self._filter_papers_by_exact_version(papers, ingested_versions)

        return LatestIngestSelection(
            papers=papers,
            fetch_stats=fetch_stats,
            migration=migration,
            skipped_after_fetch=skipped_after_fetch,
        )

    def ingest_ids(self, *, ids: list[str], skip_ingested: bool) -> ExplicitIngestSelection:
        requested_ids = self._canonical_requested_ids(ids)
        if not requested_ids:
            return ExplicitIngestSelection(
                papers=[],
                requested_ids=[],
                skipped_records=0,
                stats=ExplicitIngestStats(
                    requested_ids=0,
                    metadata_returned=0,
                    resolved_to_version=0,
                    unresolved=0,
                    skipped_existing=0,
                    to_ingest=0,
                ),
                migration=None,
            )

        ingested_versions: set[str] = set()
        migration: LegacyVersionMigrationStats | None = None
        if skip_ingested:
            migration = self._find_and_migrate_legacy_versions(self.settings)
            ingested_versions = self._load_ingested_versions(self.settings)

        papers = self._fetch_by_ids(
            ids=requested_ids,
            arxiv_api_url=self.settings.arxiv_api_url,
            timeout_seconds=self.settings.http_timeout_seconds,
            user_agent=self.settings.user_agent,
            max_retries=self.settings.arxiv_max_retries,
            backoff_start_seconds=self.settings.arxiv_backoff_start_seconds,
            backoff_cap_seconds=self.settings.arxiv_backoff_cap_seconds,
            resolve_unversioned_to_latest=skip_ingested,
        )

        stats = self._build_explicit_ingest_stats(requested_ids, papers)
        if skip_ingested:
            papers, skipped_existing = self._filter_papers_by_exact_version(papers, ingested_versions)
            stats.skipped_existing = skipped_existing
        stats.to_ingest = len(papers)

        return ExplicitIngestSelection(
            papers=papers,
            requested_ids=requested_ids,
            skipped_records=0,
            stats=stats,
            migration=migration,
        )

    def ingest_jsonl(
        self,
        *,
        source: Path,
        limit: int | None,
        skip_ingested: bool,
    ) -> ExplicitIngestSelection:
        ids, skipped_records = self._load_arxiv_ids_from_jsonl(source)

        if limit is not None:
            ids = ids[:limit]

        requested_ids = self._canonical_requested_ids(ids)
        if not requested_ids:
            return ExplicitIngestSelection(
                papers=[],
                requested_ids=[],
                skipped_records=skipped_records,
                stats=ExplicitIngestStats(
                    requested_ids=0,
                    metadata_returned=0,
                    resolved_to_version=0,
                    unresolved=0,
                    skipped_existing=0,
                    to_ingest=0,
                ),
                migration=None,
            )

        ingested_versions: set[str] = set()
        migration: LegacyVersionMigrationStats | None = None
        if skip_ingested:
            migration = self._find_and_migrate_legacy_versions(self.settings)
            ingested_versions = self._load_ingested_versions(self.settings)

        papers = self._fetch_by_ids(
            ids=requested_ids,
            arxiv_api_url=self.settings.arxiv_api_url,
            timeout_seconds=self.settings.http_timeout_seconds,
            user_agent=self.settings.user_agent,
            max_retries=self.settings.arxiv_max_retries,
            backoff_start_seconds=self.settings.arxiv_backoff_start_seconds,
            backoff_cap_seconds=self.settings.arxiv_backoff_cap_seconds,
            resolve_unversioned_to_latest=skip_ingested,
        )

        stats = self._build_explicit_ingest_stats(requested_ids, papers)
        if skip_ingested:
            papers, skipped_existing = self._filter_papers_by_exact_version(papers, ingested_versions)
            stats.skipped_existing = skipped_existing
        stats.to_ingest = len(papers)

        return ExplicitIngestSelection(
            papers=papers,
            requested_ids=requested_ids,
            skipped_records=skipped_records,
            stats=stats,
            migration=migration,
        )
