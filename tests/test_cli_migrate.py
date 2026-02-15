from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from typer.testing import CliRunner

import cv_rag.interfaces.cli.app as cli_module
from cv_rag.interfaces.cli.commands import migrate as migrate_module
from cv_rag.shared.settings import Settings


class _FakeQdrantStore:
    def __init__(self, url: str, collection_name: str) -> None:
        self.url = url
        self.collection_name = collection_name
        self.client = self

    def get_collections(self) -> object:
        class _Resp:
            def __init__(self) -> None:
                self.collections: list[object] = []

        return _Resp()

    def delete_collection_if_exists(self) -> bool:
        return True


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def test_cli_migrate_refuses_without_yes(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)

    result = runner.invoke(cli_module.app, ["migrate", "reset-reindex"])

    assert result.exit_code == 1
    assert "destructive local reset/reindex" in result.output


def test_run_migrate_reset_reindex_creates_backup_orders_steps_and_writes_report(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()

    awesome_sources = tmp_path / "awesome_sources.txt"
    visionbib_sources = tmp_path / "visionbib_sources.txt"
    dois = tmp_path / "tierA_dois.txt"
    openalex_out = tmp_path / "curation"
    backup_dir = tmp_path / "backups"

    awesome_sources.write_text("owner/repo\n", encoding="utf-8")
    visionbib_sources.write_text("https://example.com\ncompute 1 1\n", encoding="utf-8")
    dois.write_text("10.1145/3366423.3380211\n", encoding="utf-8")
    settings.sqlite_path.write_text("legacy sqlite bytes", encoding="utf-8")

    monkeypatch.setattr(
        migrate_module,
        "_preflight_checks",
        lambda **kwargs: {"ok": True, "checked": kwargs["awesome_sources"].name},
    )

    order: list[str] = []

    class _FakeIngestService:
        def __init__(self, settings: Settings) -> None:
            _ = settings
            self.calls = 0

        def list_ready_candidates(self, *, limit: int) -> list[object]:
            _ = limit
            self.calls += 1
            if self.calls == 1:
                return [object()]
            return []

        def ingest_candidates(
            self,
            *,
            candidates: list[object],
            force_grobid: bool = False,
            embed_batch_size: int | None = None,
            cache_only: bool = False,
            on_paper_progress: object | None = None,
        ) -> dict[str, int]:
            _ = (candidates, force_grobid, embed_batch_size, cache_only, on_paper_progress)
            return {
                "selected": 1,
                "queued": 1,
                "ingested": 1,
                "failed": 0,
                "blocked": 0,
            }

    def _discover_awesome(**kwargs: object) -> None:
        _ = kwargs
        order.append("discover-awesome")

    def _discover_visionbib(**kwargs: object) -> None:
        _ = kwargs
        order.append("discover-visionbib")

    def _resolve_openalex(**kwargs: object) -> None:
        _ = kwargs
        order.append("resolve-openalex")

    def _curate(**kwargs: object) -> None:
        _ = kwargs
        order.append("curate")

    report_path = migrate_module.run_migrate_reset_reindex_command(
        settings=settings,
        console=Console(record=True),
        yes=True,
        backup_dir=backup_dir,
        skip_curate=False,
        awesome_sources=awesome_sources,
        visionbib_sources=visionbib_sources,
        dois=dois,
        openalex_out_dir=openalex_out,
        ingest_batch_size=10,
        force_grobid=False,
        embed_batch_size=None,
        max_ingest_loops=10,
        run_discover_awesome_fn=_discover_awesome,
        run_discover_visionbib_fn=_discover_visionbib,
        run_resolve_openalex_fn=_resolve_openalex,
        run_curate_fn=_curate,
        ingest_service_cls=_FakeIngestService,
        qdrant_store_cls=_FakeQdrantStore,
    )

    assert order == ["discover-awesome", "discover-visionbib", "resolve-openalex", "curate"]

    backups = list(backup_dir.glob("*.bak"))
    assert backups

    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert report["backup_path"] is not None
    assert [step["name"] for step in report["steps"]] == [
        "preflight",
        "backup_sqlite",
        "reset_qdrant",
        "reset_sqlite",
        "discover_awesome",
        "discover_visionbib",
        "resolve_openalex",
        "ingest_queue",
        "curate",
    ]


def test_run_migrate_reset_reindex_is_resumable_after_failure(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()

    awesome_sources = tmp_path / "awesome_sources.txt"
    visionbib_sources = tmp_path / "visionbib_sources.txt"
    dois = tmp_path / "tierA_dois.txt"

    awesome_sources.write_text("owner/repo\n", encoding="utf-8")
    visionbib_sources.write_text("https://example.com\ncompute 1 1\n", encoding="utf-8")
    dois.write_text("10.1145/3366423.3380211\n", encoding="utf-8")

    monkeypatch.setattr(migrate_module, "_preflight_checks", lambda **kwargs: {"ok": True})

    class _NoopIngestService:
        def __init__(self, settings: Settings) -> None:
            _ = settings

        def list_ready_candidates(self, *, limit: int) -> list[object]:
            _ = limit
            return []

        def ingest_candidates(
            self,
            *,
            candidates: list[object],
            force_grobid: bool = False,
            embed_batch_size: int | None = None,
            cache_only: bool = False,
            on_paper_progress: object | None = None,
        ) -> dict[str, int]:
            _ = (candidates, force_grobid, embed_batch_size, cache_only, on_paper_progress)
            return {"selected": 0, "queued": 0, "ingested": 0, "failed": 0, "blocked": 0}

    calls = {"n": 0}

    def flaky_resolve(**kwargs: object) -> None:
        _ = kwargs
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("temporary resolver failure")

    kwargs = dict(
        settings=settings,
        console=Console(record=True),
        yes=True,
        backup_dir=None,
        skip_curate=True,
        awesome_sources=awesome_sources,
        visionbib_sources=visionbib_sources,
        dois=dois,
        openalex_out_dir=tmp_path / "curation",
        ingest_batch_size=10,
        force_grobid=False,
        embed_batch_size=None,
        max_ingest_loops=5,
        run_discover_awesome_fn=lambda **_: None,
        run_discover_visionbib_fn=lambda **_: None,
        run_resolve_openalex_fn=flaky_resolve,
        run_curate_fn=lambda **_: None,
        ingest_service_cls=_NoopIngestService,
        qdrant_store_cls=_FakeQdrantStore,
    )

    try:
        migrate_module.run_migrate_reset_reindex_command(**kwargs)
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:
        raise AssertionError("expected first migration run to fail")

    report_dir = settings.data_dir / "migrations"
    failed_reports = sorted(report_dir.glob("*-reset-reindex-report.json"))
    assert failed_reports
    failed_report = json.loads(failed_reports[-1].read_text(encoding="utf-8"))
    assert failed_report["status"] == "fail"

    report_path = migrate_module.run_migrate_reset_reindex_command(**kwargs)
    success_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert success_report["status"] == "ok"


def test_run_migrate_reset_reindex_cache_only_restores_from_run_artifacts(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()
    runs_dir = settings.discovery_runs_dir
    runs_dir.mkdir(parents=True, exist_ok=True)

    awesome_run = runs_dir / "20260215T010000Z-awesome"
    visionbib_run = runs_dir / "20260215T010000Z-visionbib"
    openalex_run = runs_dir / "20260215T010000Z-openalex"
    awesome_run.mkdir(parents=True, exist_ok=True)
    visionbib_run.mkdir(parents=True, exist_ok=True)
    openalex_run.mkdir(parents=True, exist_ok=True)

    (awesome_run / "awesome_references.jsonl").write_text(
        json.dumps(
            {
                "ref_type": "arxiv",
                "normalized_value": "2104.00680v2",
                "source_kind": "curated_repo",
                "source_ref": "https://github.com/org/repo",
                "discovered_at_unix": 1700000000,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (visionbib_run / "visionbib_references.jsonl").write_text(
        json.dumps(
            {
                "ref_type": "pdf_url",
                "normalized_value": "https://example.org/sample.pdf",
                "source_kind": "curated_repo",
                "source_ref": "https://visionbib.com",
                "discovered_at_unix": 1700000001,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (openalex_run / "openalex_resolution.jsonl").write_text(
        json.dumps(
            {
                "doc_id": "axv:2104.00680v2",
                "arxiv_id": "2104.00680",
                "arxiv_id_with_version": "2104.00680v2",
                "doi": "10.1000/demo",
                "pdf_url": "https://arxiv.org/pdf/2104.00680v2.pdf",
                "resolution_confidence": 0.95,
                "source_kind": "openalex_resolved",
                "resolved_at_unix": 1700000002,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(migrate_module, "_preflight_checks", lambda **kwargs: {"ok": True})

    class _NoopIngestService:
        def __init__(self, settings: Settings) -> None:
            _ = settings

        def list_ready_candidates(self, *, limit: int) -> list[object]:
            _ = limit
            return []

        def ingest_candidates(
            self,
            *,
            candidates: list[object],
            force_grobid: bool = False,
            embed_batch_size: int | None = None,
            cache_only: bool = False,
            on_paper_progress: object | None = None,
        ) -> dict[str, int]:
            _ = (candidates, force_grobid, embed_batch_size, cache_only, on_paper_progress)
            return {"selected": 0, "queued": 0, "ingested": 0, "failed": 0, "blocked": 0}

    def _should_not_be_called(**kwargs: object) -> None:
        _ = kwargs
        raise AssertionError("network discovery/resolve should not run in cache-only mode")

    report_path = migrate_module.run_migrate_reset_reindex_command(
        settings=settings,
        console=Console(record=True),
        yes=True,
        backup_dir=None,
        skip_curate=False,
        cache_only=True,
        awesome_sources=tmp_path / "unused-awesome.txt",
        visionbib_sources=tmp_path / "unused-visionbib.txt",
        dois=tmp_path / "unused-dois.txt",
        openalex_out_dir=tmp_path / "unused-openalex",
        cache_awesome_refs=awesome_run / "awesome_references.jsonl",
        cache_visionbib_refs=visionbib_run / "visionbib_references.jsonl",
        cache_openalex_resolution=openalex_run / "openalex_resolution.jsonl",
        ingest_batch_size=10,
        force_grobid=False,
        embed_batch_size=None,
        max_ingest_loops=10,
        run_discover_awesome_fn=_should_not_be_called,
        run_discover_visionbib_fn=_should_not_be_called,
        run_resolve_openalex_fn=_should_not_be_called,
        run_curate_fn=_should_not_be_called,
        ingest_service_cls=_NoopIngestService,
        qdrant_store_cls=_FakeQdrantStore,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "ok"
    assert [step["name"] for step in report["steps"]] == [
        "preflight",
        "backup_sqlite",
        "reset_qdrant",
        "reset_sqlite",
        "restore_cached_references",
        "ingest_queue",
        "curate",
    ]
    assert report["steps"][-1]["details"]["reason"] == "cache_only_mode"

    restored = next(step for step in report["steps"] if step["name"] == "restore_cached_references")
    assert int(restored["details"]["refs_loaded"]) == 2
    assert int(restored["details"]["resolved_loaded"]) == 1
