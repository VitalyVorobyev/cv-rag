from __future__ import annotations

import time
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
from cv_rag.ingest.pdf_pipeline import IngestPipeline
from cv_rag.shared.settings import Settings
from cv_rag.storage.repositories import IngestCandidate
from cv_rag.storage.sqlite import SQLiteStore


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


def list_ready_candidates(*, sqlite_store: SQLiteStore, limit: int) -> list[IngestCandidate]:
    """Return ready candidates ranked by queue priority."""
    return sqlite_store.list_ready_candidates(limit=limit)


def mark_candidate_result(
    *,
    sqlite_store: SQLiteStore,
    settings: Settings,
    doc_id: str,
    status: str,
    reason: str | None = None,
) -> None:
    """Persist a candidate status transition."""
    sqlite_store.mark_candidate_result(
        doc_id=doc_id,
        status=status,
        reason=reason,
        candidate_retry_days=settings.candidate_retry_days,
        candidate_max_retries=settings.candidate_max_retries,
    )


def ingest_candidates(
    *,
    settings: Settings,
    candidates: list[IngestCandidate],
    force_grobid: bool = False,
    embed_batch_size: int | None = None,
) -> dict[str, int]:
    """Ingest queued candidates with available fulltext."""
    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.create_schema()
    try:
        docs = sqlite_store.get_documents_by_ids([c.doc_id for c in candidates])
        by_doc_id = {str(doc["doc_id"]): doc for doc in docs}
        papers: list[PaperMetadata] = []
        blocked = 0

        for candidate in candidates:
            doc = by_doc_id.get(candidate.doc_id, {})
            arxiv_id_with_version = str(doc.get("arxiv_id_with_version") or "").strip()
            arxiv_id = str(doc.get("arxiv_id") or "").strip()
            provenance_kind = str(doc.get("provenance_kind") or "").strip() or None
            pdf_url = (candidate.best_pdf_url or str(doc.get("pdf_url") or "").strip() or None)

            if not pdf_url:
                mark_candidate_result(
                    sqlite_store=sqlite_store,
                    settings=settings,
                    doc_id=candidate.doc_id,
                    status="blocked",
                    reason="missing_pdf_url",
                )
                blocked += 1
                continue

            if not arxiv_id_with_version:
                arxiv_id_with_version = candidate.doc_id
            if not arxiv_id:
                arxiv_id = arxiv_id_with_version.split("v", 1)[0]
            abs_url = (
                f"https://arxiv.org/abs/{arxiv_id_with_version}"
                if ":" not in arxiv_id
                else pdf_url
            )

            papers.append(
                PaperMetadata(
                    arxiv_id=arxiv_id,
                    arxiv_id_with_version=arxiv_id_with_version,
                    version=None,
                    doc_id=candidate.doc_id,
                    provenance_kind=provenance_kind,
                    title=str(doc.get("doi") or candidate.doc_id),
                    summary="",
                    published=None,
                    updated=None,
                    authors=[],
                    pdf_url=pdf_url,
                    abs_url=abs_url,
                )
            )

        if not papers:
            return {
                "selected": len(candidates),
                "queued": 0,
                "ingested": 0,
                "failed": 0,
                "blocked": blocked,
            }

        pipeline = IngestPipeline(settings)
        metadata_path = settings.metadata_dir / f"corpus_candidates_{int(time.time())}.json"
        result = pipeline.run(
            papers=papers,
            metadata_json_path=metadata_path,
            force_grobid=force_grobid,
            embed_batch_size=embed_batch_size,
        )
        failed_prefixes: set[str] = set()
        for failure in result.failed_papers:
            if ": " in failure:
                failed_prefixes.add(failure.split(": ", 1)[0].strip())
                continue
            if ":" in failure:
                failed_prefixes.add(failure.rsplit(":", 1)[0].strip())
        ingested = 0
        failed = 0
        for paper in papers:
            if paper.arxiv_id in failed_prefixes:
                failed += 1
                mark_candidate_result(
                    sqlite_store=sqlite_store,
                    settings=settings,
                    doc_id=paper.resolved_doc_id(),
                    status="failed",
                    reason="pipeline_failed",
                )
            else:
                ingested += 1
                mark_candidate_result(
                    sqlite_store=sqlite_store,
                    settings=settings,
                    doc_id=paper.resolved_doc_id(),
                    status="ingested",
                )
        return {
            "selected": len(candidates),
            "queued": len(papers),
            "ingested": ingested,
            "failed": failed,
            "blocked": blocked,
        }
    finally:
        sqlite_store.close()


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

    def list_ready_candidates(self, *, limit: int) -> list[IngestCandidate]:
        sqlite_store = SQLiteStore(self.settings.sqlite_path)
        sqlite_store.create_schema()
        try:
            return list_ready_candidates(sqlite_store=sqlite_store, limit=limit)
        finally:
            sqlite_store.close()

    def ingest_candidates(
        self,
        *,
        candidates: list[IngestCandidate],
        force_grobid: bool = False,
        embed_batch_size: int | None = None,
    ) -> dict[str, int]:
        return ingest_candidates(
            settings=self.settings,
            candidates=candidates,
            force_grobid=force_grobid,
            embed_batch_size=embed_batch_size,
        )
