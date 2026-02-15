from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from cv_rag.ingest.dedupe import ExplicitIngestStats, LegacyVersionMigrationStats
from cv_rag.ingest.models import PaperMetadata
from cv_rag.ingest.service import IngestService
from cv_rag.shared.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def _paper(arxiv_id_with_version: str, *, title: str = "Paper", fused_id: str | None = None) -> PaperMetadata:
    base_id = fused_id or arxiv_id_with_version.split("v", 1)[0]
    version = None
    if "v" in arxiv_id_with_version:
        version = f"v{arxiv_id_with_version.rsplit('v', 1)[1]}"
    return PaperMetadata(
        arxiv_id=base_id,
        arxiv_id_with_version=arxiv_id_with_version,
        version=version,
        title=title,
        summary="summary",
        published=None,
        updated=None,
        authors=["A. Author"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id_with_version}.pdf",
        abs_url=f"https://arxiv.org/abs/{arxiv_id_with_version}",
    )


def test_ingest_latest_without_skip_bypasses_migration_and_filter(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}
    papers = [_paper("2602.00001v1")]

    def fake_fetch_latest(
        *,
        limit: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        skip_arxiv_id_with_version: set[str] | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[PaperMetadata]:
        _ = (arxiv_api_url, timeout_seconds, user_agent, max_retries, backoff_start_seconds, backoff_cap_seconds)
        captured["limit"] = limit
        captured["skip_set"] = skip_arxiv_id_with_version
        if stats is not None:
            stats.update({"requested": limit, "selected": len(papers)})
        return papers

    def fail_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        _ = current_settings
        raise AssertionError("migration hook must not run when skip_ingested=False")

    def fail_load_versions(current_settings: Settings) -> set[str]:
        _ = current_settings
        raise AssertionError("version loading must not run when skip_ingested=False")

    def fail_filter(
        candidate_papers: list[PaperMetadata],
        ingested_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        _ = (candidate_papers, ingested_versions)
        raise AssertionError("exact-version filter must not run when skip_ingested=False")

    service = IngestService(
        settings,
        fetch_latest_fn=fake_fetch_latest,
        find_and_migrate_legacy_versions_fn=fail_migration,
        load_ingested_versions_fn=fail_load_versions,
        filter_papers_by_exact_version_fn=fail_filter,
    )

    result = service.ingest_latest(limit=3, skip_ingested=False)

    assert captured["limit"] == 3
    assert captured["skip_set"] is None
    assert [paper.arxiv_id_with_version for paper in result.papers] == ["2602.00001v1"]
    assert result.fetch_stats == {"requested": 3, "selected": 1}
    assert result.migration is None
    assert result.skipped_after_fetch == 0


def test_ingest_latest_with_skip_runs_migration_load_and_filter(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}
    migration_stats = LegacyVersionMigrationStats(legacy_rows_found=1, migrated=1, unresolved=0)
    ingested_versions = {"2602.00001v1"}
    fetched_papers = [_paper("2602.00001v1"), _paper("2602.00002v1")]

    def fake_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        assert current_settings == settings
        captured["migration_called"] = True
        return migration_stats

    def fake_load_versions(current_settings: Settings) -> set[str]:
        assert current_settings == settings
        captured["load_called"] = True
        return ingested_versions

    def fake_fetch_latest(
        *,
        limit: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        skip_arxiv_id_with_version: set[str] | None = None,
        stats: dict[str, int] | None = None,
    ) -> list[PaperMetadata]:
        _ = (
            arxiv_api_url,
            timeout_seconds,
            user_agent,
            max_retries,
            backoff_start_seconds,
            backoff_cap_seconds,
        )
        captured["limit"] = limit
        captured["skip_set"] = skip_arxiv_id_with_version
        if stats is not None:
            stats.update({"requested": limit, "selected": len(fetched_papers)})
        return fetched_papers

    def fake_filter(
        candidate_papers: list[PaperMetadata],
        known_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        captured["filter_called"] = True
        assert [paper.arxiv_id_with_version for paper in candidate_papers] == [
            "2602.00001v1",
            "2602.00002v1",
        ]
        assert known_versions == ingested_versions
        return [candidate_papers[1]], 1

    service = IngestService(
        settings,
        fetch_latest_fn=fake_fetch_latest,
        find_and_migrate_legacy_versions_fn=fake_migration,
        load_ingested_versions_fn=fake_load_versions,
        filter_papers_by_exact_version_fn=fake_filter,
    )

    result = service.ingest_latest(limit=2, skip_ingested=True)

    assert captured["migration_called"] is True
    assert captured["load_called"] is True
    assert captured["filter_called"] is True
    assert captured["limit"] == 2
    assert captured["skip_set"] == ingested_versions
    assert result.fetch_stats == {"requested": 2, "selected": 2}
    assert result.migration == migration_stats
    assert [paper.arxiv_id_with_version for paper in result.papers] == ["2602.00002v1"]
    assert result.skipped_after_fetch == 1


def test_ingest_ids_returns_empty_when_canonical_ids_empty(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}

    def fake_canonical(raw_ids: list[str]) -> list[str]:
        captured["raw_ids"] = raw_ids
        return []

    def fail_fetch_by_ids(**kwargs: object) -> list[PaperMetadata]:
        _ = kwargs
        raise AssertionError("fetch_by_ids must not be called for empty canonical IDs")

    def fail_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        _ = current_settings
        raise AssertionError("migration must not run for empty canonical IDs")

    def fail_build_stats(requested_ids: list[str], papers: list[PaperMetadata]) -> ExplicitIngestStats:
        _ = (requested_ids, papers)
        raise AssertionError("stats builder must not run for empty canonical IDs")

    service = IngestService(
        settings,
        fetch_by_ids_fn=fail_fetch_by_ids,
        find_and_migrate_legacy_versions_fn=fail_migration,
        canonical_requested_ids_fn=fake_canonical,
        build_explicit_ingest_stats_fn=fail_build_stats,
    )

    result = service.ingest_ids(ids=["", "invalid"], skip_ingested=True)

    assert captured["raw_ids"] == ["", "invalid"]
    assert result.requested_ids == []
    assert result.papers == []
    assert result.skipped_records == 0
    assert result.migration is None
    assert result.stats.requested_ids == 0
    assert result.stats.metadata_returned == 0
    assert result.stats.resolved_to_version == 0
    assert result.stats.unresolved == 0
    assert result.stats.skipped_existing == 0
    assert result.stats.to_ingest == 0


def test_ingest_ids_without_skip_avoids_filter_and_sets_to_ingest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}
    canonical_ids = ["2104.00680", "1911.11763v2"]
    fetched_papers = [_paper("2104.00680v3"), _paper("1911.11763v2")]

    def fake_canonical(raw_ids: list[str]) -> list[str]:
        captured["raw_ids"] = raw_ids
        return canonical_ids

    def fake_fetch_by_ids(
        *,
        ids: list[str],
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        resolve_unversioned_to_latest: bool,
    ) -> list[PaperMetadata]:
        _ = (arxiv_api_url, timeout_seconds, user_agent, max_retries, backoff_start_seconds, backoff_cap_seconds)
        captured["fetch_ids"] = ids
        captured["resolve"] = resolve_unversioned_to_latest
        return fetched_papers

    def fake_build_stats(requested_ids: list[str], papers: list[PaperMetadata]) -> ExplicitIngestStats:
        captured["stats_requested_ids"] = requested_ids
        captured["stats_papers"] = papers
        return ExplicitIngestStats(
            requested_ids=len(requested_ids),
            metadata_returned=len(papers),
            resolved_to_version=1,
            unresolved=0,
        )

    def fail_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        _ = current_settings
        raise AssertionError("migration must not run when skip_ingested=False")

    def fail_load_versions(current_settings: Settings) -> set[str]:
        _ = current_settings
        raise AssertionError("load_ingested_versions must not run when skip_ingested=False")

    def fail_filter(
        candidate_papers: list[PaperMetadata],
        known_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        _ = (candidate_papers, known_versions)
        raise AssertionError("exact-version filter must not run when skip_ingested=False")

    service = IngestService(
        settings,
        fetch_by_ids_fn=fake_fetch_by_ids,
        find_and_migrate_legacy_versions_fn=fail_migration,
        load_ingested_versions_fn=fail_load_versions,
        canonical_requested_ids_fn=fake_canonical,
        filter_papers_by_exact_version_fn=fail_filter,
        build_explicit_ingest_stats_fn=fake_build_stats,
    )

    result = service.ingest_ids(ids=[" 2104.00680 ", "1911.11763v2"], skip_ingested=False)

    assert captured["raw_ids"] == [" 2104.00680 ", "1911.11763v2"]
    assert captured["fetch_ids"] == canonical_ids
    assert captured["resolve"] is False
    assert captured["stats_requested_ids"] == canonical_ids
    assert [paper.arxiv_id_with_version for paper in captured["stats_papers"]] == [
        "2104.00680v3",
        "1911.11763v2",
    ]
    assert result.requested_ids == canonical_ids
    assert [paper.arxiv_id_with_version for paper in result.papers] == ["2104.00680v3", "1911.11763v2"]
    assert result.stats.skipped_existing == 0
    assert result.stats.to_ingest == 2
    assert result.migration is None


def test_ingest_ids_with_skip_runs_migration_fetch_and_filter(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}
    canonical_ids = ["2104.00680", "1911.11763"]
    migration_stats = LegacyVersionMigrationStats(legacy_rows_found=2, migrated=2, unresolved=0)
    ingested_versions = {"2104.00680v3"}
    fetched_papers = [_paper("2104.00680v3"), _paper("1911.11763v2")]

    def fake_canonical(raw_ids: list[str]) -> list[str]:
        captured["raw_ids"] = raw_ids
        return canonical_ids

    def fake_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        assert current_settings == settings
        captured["migration_called"] = True
        return migration_stats

    def fake_load_versions(current_settings: Settings) -> set[str]:
        assert current_settings == settings
        captured["load_called"] = True
        return ingested_versions

    def fake_fetch_by_ids(
        *,
        ids: list[str],
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        resolve_unversioned_to_latest: bool,
    ) -> list[PaperMetadata]:
        _ = (arxiv_api_url, timeout_seconds, user_agent, max_retries, backoff_start_seconds, backoff_cap_seconds)
        captured["fetch_ids"] = ids
        captured["resolve"] = resolve_unversioned_to_latest
        return fetched_papers

    def fake_build_stats(requested_ids: list[str], papers: list[PaperMetadata]) -> ExplicitIngestStats:
        _ = papers
        return ExplicitIngestStats(
            requested_ids=len(requested_ids),
            metadata_returned=2,
            resolved_to_version=1,
            unresolved=0,
        )

    def fake_filter(
        candidate_papers: list[PaperMetadata],
        known_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        captured["filter_called"] = True
        assert known_versions == ingested_versions
        assert [paper.arxiv_id_with_version for paper in candidate_papers] == [
            "2104.00680v3",
            "1911.11763v2",
        ]
        return [candidate_papers[1]], 1

    service = IngestService(
        settings,
        fetch_by_ids_fn=fake_fetch_by_ids,
        find_and_migrate_legacy_versions_fn=fake_migration,
        load_ingested_versions_fn=fake_load_versions,
        canonical_requested_ids_fn=fake_canonical,
        filter_papers_by_exact_version_fn=fake_filter,
        build_explicit_ingest_stats_fn=fake_build_stats,
    )

    result = service.ingest_ids(ids=["2104.00680", "1911.11763"], skip_ingested=True)

    assert captured["raw_ids"] == ["2104.00680", "1911.11763"]
    assert captured["migration_called"] is True
    assert captured["load_called"] is True
    assert captured["fetch_ids"] == canonical_ids
    assert captured["resolve"] is True
    assert captured["filter_called"] is True
    assert result.requested_ids == canonical_ids
    assert [paper.arxiv_id_with_version for paper in result.papers] == ["1911.11763v2"]
    assert result.stats.skipped_existing == 1
    assert result.stats.to_ingest == 1
    assert result.migration == migration_stats


@pytest.mark.parametrize("skip_ingested", [False, True])
def test_ingest_jsonl_honors_limit_and_matches_skip_behavior(
    skip_ingested: bool,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    captured: dict[str, Any] = {}
    migration_stats = LegacyVersionMigrationStats(legacy_rows_found=1, migrated=1, unresolved=0)
    ingested_versions = {"2104.00680v3"}

    def fake_load_jsonl(source: Path) -> tuple[list[str], int]:
        captured["source"] = source
        return ["2104.00680", "1911.11763", "2104.00680"], 2

    def fake_canonical(raw_ids: list[str]) -> list[str]:
        captured["canonical_input"] = raw_ids
        return ["2104.00680", "1911.11763"]

    def fake_fetch_by_ids(
        *,
        ids: list[str],
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        resolve_unversioned_to_latest: bool,
    ) -> list[PaperMetadata]:
        _ = (arxiv_api_url, timeout_seconds, user_agent, max_retries, backoff_start_seconds, backoff_cap_seconds)
        captured["fetch_ids"] = ids
        captured["resolve"] = resolve_unversioned_to_latest
        return [_paper("2104.00680v3"), _paper("1911.11763v2")]

    def fake_build_stats(requested_ids: list[str], papers: list[PaperMetadata]) -> ExplicitIngestStats:
        _ = papers
        return ExplicitIngestStats(
            requested_ids=len(requested_ids),
            metadata_returned=2,
            resolved_to_version=1,
            unresolved=0,
        )

    def fake_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        assert current_settings == settings
        captured["migration_called"] = True
        return migration_stats

    def fake_load_versions(current_settings: Settings) -> set[str]:
        assert current_settings == settings
        captured["load_called"] = True
        return ingested_versions

    def fake_filter(
        candidate_papers: list[PaperMetadata],
        known_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        captured["filter_called"] = True
        assert known_versions == ingested_versions
        return [paper for paper in candidate_papers if paper.arxiv_id_with_version != "2104.00680v3"], 1

    def fail_migration(current_settings: Settings) -> LegacyVersionMigrationStats:
        _ = current_settings
        raise AssertionError("migration must not run when skip_ingested=False")

    def fail_load_versions(current_settings: Settings) -> set[str]:
        _ = current_settings
        raise AssertionError("version loading must not run when skip_ingested=False")

    def fail_filter(
        candidate_papers: list[PaperMetadata],
        known_versions: set[str],
    ) -> tuple[list[PaperMetadata], int]:
        _ = (candidate_papers, known_versions)
        raise AssertionError("exact-version filter must not run when skip_ingested=False")

    service = IngestService(
        settings,
        fetch_by_ids_fn=fake_fetch_by_ids,
        find_and_migrate_legacy_versions_fn=fake_migration if skip_ingested else fail_migration,
        load_ingested_versions_fn=fake_load_versions if skip_ingested else fail_load_versions,
        canonical_requested_ids_fn=fake_canonical,
        filter_papers_by_exact_version_fn=fake_filter if skip_ingested else fail_filter,
        build_explicit_ingest_stats_fn=fake_build_stats,
        load_arxiv_ids_from_jsonl_fn=fake_load_jsonl,
    )

    source = tmp_path / "seed.jsonl"
    result = service.ingest_jsonl(source=source, limit=2, skip_ingested=skip_ingested)

    assert captured["source"] == source
    assert captured["canonical_input"] == ["2104.00680", "1911.11763"]
    assert captured["fetch_ids"] == ["2104.00680", "1911.11763"]
    assert captured["resolve"] is skip_ingested
    assert result.skipped_records == 2
    assert result.requested_ids == ["2104.00680", "1911.11763"]
    if skip_ingested:
        assert captured["migration_called"] is True
        assert captured["load_called"] is True
        assert captured["filter_called"] is True
        assert [paper.arxiv_id_with_version for paper in result.papers] == ["1911.11763v2"]
        assert result.stats.skipped_existing == 1
        assert result.stats.to_ingest == 1
        assert result.migration == migration_stats
    else:
        assert [paper.arxiv_id_with_version for paper in result.papers] == ["2104.00680v3", "1911.11763v2"]
        assert result.stats.skipped_existing == 0
        assert result.stats.to_ingest == 2
        assert result.migration is None


@pytest.mark.parametrize(
    "raised_error",
    [
        FileNotFoundError("JSONL file not found"),
        ValueError("invalid JSON"),
    ],
)
def test_ingest_jsonl_propagates_loader_errors(
    raised_error: Exception,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)

    def fake_load_jsonl(source: Path) -> tuple[list[str], int]:
        _ = source
        raise raised_error

    service = IngestService(
        settings,
        load_arxiv_ids_from_jsonl_fn=fake_load_jsonl,
    )

    with pytest.raises(type(raised_error)) as exc_info:
        service.ingest_jsonl(source=tmp_path / "missing.jsonl", limit=None, skip_ingested=False)

    assert str(exc_info.value) == str(raised_error)
