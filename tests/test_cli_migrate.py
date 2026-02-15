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
        ) -> dict[str, int]:
            _ = (candidates, force_grobid, embed_batch_size)
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
        ) -> dict[str, int]:
            _ = (candidates, force_grobid, embed_batch_size)
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
