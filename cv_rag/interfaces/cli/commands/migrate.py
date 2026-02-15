from __future__ import annotations

import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import typer
from rich.console import Console

from cv_rag.curation.s2_client import SemanticScholarClient
from cv_rag.ingest.service import IngestService
from cv_rag.interfaces.cli.commands.corpus import (
    run_corpus_discover_awesome_command,
    run_corpus_discover_visionbib_command,
    run_corpus_resolve_openalex_command,
)
from cv_rag.interfaces.cli.commands.curate import run_curate_command
from cv_rag.shared.settings import Settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore


@dataclass(slots=True)
class MigrationStep:
    name: str
    status: str
    duration_seconds: float
    details: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class MigrationReport:
    run_id: str
    started_at_utc: str
    finished_at_utc: str | None
    status: str
    sqlite_path: str
    qdrant_collection: str
    backup_path: str | None
    report_path: str
    steps: list[MigrationStep]


def _utc_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _bool_service_ok(url: str, *, timeout_seconds: float = 5.0) -> bool:
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return False
    return True


def _preflight_checks(
    *,
    settings: Settings,
    awesome_sources: Path,
    visionbib_sources: Path,
    dois: Path,
    qdrant_store_cls: type[QdrantStore],
) -> dict[str, object]:
    missing_inputs: list[str] = []
    for path in (awesome_sources, visionbib_sources, dois):
        if not path.exists():
            missing_inputs.append(str(path))
    if missing_inputs:
        raise FileNotFoundError(
            "Preflight failed; missing required input files: " + ", ".join(missing_inputs)
        )

    qdrant_store = qdrant_store_cls(settings.qdrant_url, settings.qdrant_collection)
    qdrant_store.client.get_collections()

    grobid_ok = _bool_service_ok(f"{settings.grobid_url.rstrip('/')}/api/isalive")
    ollama_ok = _bool_service_ok(f"{settings.ollama_url.rstrip('/')}/api/version")
    qdrant_ok = _bool_service_ok(settings.qdrant_url)

    if not qdrant_ok:
        raise RuntimeError(f"Preflight failed; Qdrant is unreachable: {settings.qdrant_url}")
    if not grobid_ok:
        raise RuntimeError(f"Preflight failed; GROBID is unreachable: {settings.grobid_url}")
    if not ollama_ok:
        raise RuntimeError(f"Preflight failed; Ollama is unreachable: {settings.ollama_url}")

    return {
        "awesome_sources": str(awesome_sources),
        "visionbib_sources": str(visionbib_sources),
        "dois": str(dois),
        "services": {
            "qdrant": qdrant_ok,
            "grobid": grobid_ok,
            "ollama": ollama_ok,
        },
    }


def _backup_sqlite(*, sqlite_path: Path, backup_dir: Path | None, run_id: str) -> Path | None:
    if backup_dir is None or not sqlite_path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{run_id}-{sqlite_path.name}.bak"
    shutil.copy2(sqlite_path, target)

    wal_file = sqlite_path.with_suffix(sqlite_path.suffix + "-wal")
    shm_file = sqlite_path.with_suffix(sqlite_path.suffix + "-shm")
    if wal_file.exists():
        shutil.copy2(wal_file, backup_dir / f"{run_id}-{wal_file.name}.bak")
    if shm_file.exists():
        shutil.copy2(shm_file, backup_dir / f"{run_id}-{shm_file.name}.bak")

    return target


def _reset_sqlite(*, sqlite_path: Path, sqlite_store_cls: type[SQLiteStore]) -> dict[str, object]:
    removed: list[str] = []
    for path in (
        sqlite_path,
        sqlite_path.with_suffix(sqlite_path.suffix + "-wal"),
        sqlite_path.with_suffix(sqlite_path.suffix + "-shm"),
    ):
        if path.exists():
            path.unlink()
            removed.append(str(path))

    sqlite_store = sqlite_store_cls(sqlite_path)
    try:
        sqlite_store.create_schema()
    finally:
        sqlite_store.close()

    return {"removed": removed, "sqlite": str(sqlite_path)}


def _reset_qdrant(*, settings: Settings, qdrant_store_cls: type[QdrantStore]) -> dict[str, object]:
    qdrant_store = qdrant_store_cls(settings.qdrant_url, settings.qdrant_collection)
    deleted = qdrant_store.delete_collection_if_exists()
    return {"collection": settings.qdrant_collection, "deleted": deleted}


def _run_ingest_queue(
    *,
    settings: Settings,
    ingest_batch_size: int,
    force_grobid: bool,
    embed_batch_size: int | None,
    max_ingest_loops: int,
    ingest_service_cls: type[IngestService],
) -> dict[str, object]:
    service = ingest_service_cls(settings)

    loops = 0
    selected_total = 0
    queued_total = 0
    ingested_total = 0
    failed_total = 0
    blocked_total = 0

    while True:
        loops += 1
        if loops > max_ingest_loops:
            raise RuntimeError(
                f"Ingest loop exceeded max iterations ({max_ingest_loops}); queue may be unstable"
            )

        candidates = service.list_ready_candidates(limit=ingest_batch_size)
        if not candidates:
            break

        stats = service.ingest_candidates(
            candidates=candidates,
            force_grobid=force_grobid,
            embed_batch_size=embed_batch_size,
        )
        selected_total += int(stats.get("selected", 0))
        queued_total += int(stats.get("queued", 0))
        ingested_total += int(stats.get("ingested", 0))
        failed_total += int(stats.get("failed", 0))
        blocked_total += int(stats.get("blocked", 0))

    return {
        "loops": loops - 1,
        "selected": selected_total,
        "queued": queued_total,
        "ingested": ingested_total,
        "failed": failed_total,
        "blocked": blocked_total,
    }


def _write_report(report: MigrationReport, report_path: Path) -> None:
    payload = asdict(report)
    payload["steps"] = [asdict(step) for step in report.steps]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_migrate_reset_reindex_command(
    *,
    settings: Settings,
    console: Console,
    yes: bool,
    backup_dir: Path | None,
    skip_curate: bool,
    awesome_sources: Path,
    visionbib_sources: Path,
    dois: Path,
    openalex_out_dir: Path,
    ingest_batch_size: int,
    force_grobid: bool,
    embed_batch_size: int | None,
    max_ingest_loops: int,
    run_discover_awesome_fn: object = run_corpus_discover_awesome_command,
    run_discover_visionbib_fn: object = run_corpus_discover_visionbib_command,
    run_resolve_openalex_fn: object = run_corpus_resolve_openalex_command,
    run_curate_fn: object = run_curate_command,
    ingest_service_cls: type[IngestService] = IngestService,
    qdrant_store_cls: type[QdrantStore] = QdrantStore,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
    s2_client_cls: type[SemanticScholarClient] = SemanticScholarClient,
) -> Path:
    if not yes:
        console.print(
            "[red]This command performs destructive local reset/reindex. Re-run with --yes to continue.[/red]"
        )
        raise typer.Exit(code=1)

    settings.ensure_directories()
    run_id = _utc_run_id()
    report_path = settings.data_dir / "migrations" / f"{run_id}-reset-reindex-report.json"
    steps: list[MigrationStep] = []
    backup_path: Path | None = None
    started_at = datetime.now(UTC)
    status = "ok"

    def _run_step(name: str, fn: object) -> dict[str, object]:
        t0 = time.perf_counter()
        try:
            details = fn()  # type: ignore[operator]
            if details is None:
                details = {}
            details_dict = dict(details)
            steps.append(
                MigrationStep(
                    name=name,
                    status="ok",
                    duration_seconds=round(time.perf_counter() - t0, 3),
                    details=details_dict,
                )
            )
            return details_dict
        except Exception as exc:
            steps.append(
                MigrationStep(
                    name=name,
                    status="fail",
                    duration_seconds=round(time.perf_counter() - t0, 3),
                    details={"error": str(exc)},
                )
            )
            raise

    try:
        _run_step(
            "preflight",
            lambda: _preflight_checks(
                settings=settings,
                awesome_sources=awesome_sources,
                visionbib_sources=visionbib_sources,
                dois=dois,
                qdrant_store_cls=qdrant_store_cls,
            ),
        )

        backup_details = _run_step(
            "backup_sqlite",
            lambda: {
                "backup": str(
                    _backup_sqlite(
                        sqlite_path=settings.sqlite_path,
                        backup_dir=backup_dir,
                        run_id=run_id,
                    )
                    or ""
                )
            },
        )
        backup_path_str = str(backup_details.get("backup", "")).strip()
        backup_path = Path(backup_path_str) if backup_path_str else None

        _run_step(
            "reset_qdrant",
            lambda: _reset_qdrant(settings=settings, qdrant_store_cls=qdrant_store_cls),
        )
        _run_step(
            "reset_sqlite",
            lambda: _reset_sqlite(sqlite_path=settings.sqlite_path, sqlite_store_cls=sqlite_store_cls),
        )

        _run_step(
            "discover_awesome",
            lambda: run_discover_awesome_fn(  # type: ignore[operator]
                settings=settings,
                console=console,
                sources=awesome_sources,
                run_id=f"{run_id}-awesome",
            ),
        )
        _run_step(
            "discover_visionbib",
            lambda: run_discover_visionbib_fn(  # type: ignore[operator]
                settings=settings,
                console=console,
                sources=visionbib_sources,
                run_id=f"{run_id}-visionbib",
            ),
        )
        _run_step(
            "resolve_openalex",
            lambda: run_resolve_openalex_fn(  # type: ignore[operator]
                settings=settings,
                console=console,
                dois=dois,
                out_dir=openalex_out_dir,
                run_id=f"{run_id}-openalex",
                email=None,
                api_key=os.getenv("OPENALEX_API_KEY"),
            ),
        )

        _run_step(
            "ingest_queue",
            lambda: _run_ingest_queue(
                settings=settings,
                ingest_batch_size=ingest_batch_size,
                force_grobid=force_grobid,
                embed_batch_size=embed_batch_size,
                max_ingest_loops=max_ingest_loops,
                ingest_service_cls=ingest_service_cls,
            ),
        )

        if skip_curate:
            _run_step("curate", lambda: {"skipped": True})
        else:
            _run_step(
                "curate",
                lambda: run_curate_fn(  # type: ignore[operator]
                    settings=settings,
                    console=console,
                    refresh_days=30,
                    tier0_venues=Path("data/venues_tier0.txt"),
                    tier0_min_citations=200,
                    tier0_min_cpy=30.0,
                    tier1_min_citations=20,
                    tier1_min_cpy=3.0,
                    limit=None,
                    sqlite_store_cls=sqlite_store_cls,
                    s2_client_cls=s2_client_cls,
                ),
            )

    except Exception as exc:  # noqa: BLE001
        status = "fail"
        console.print(f"[red]Migration failed: {exc}[/red]")
        console.print("[yellow]You can rerun the command; it is designed for reset/retry workflows.[/yellow]")
    finally:
        finished_at = datetime.now(UTC)
        report = MigrationReport(
            run_id=run_id,
            started_at_utc=started_at.isoformat(),
            finished_at_utc=finished_at.isoformat(),
            status=status,
            sqlite_path=str(settings.sqlite_path),
            qdrant_collection=settings.qdrant_collection,
            backup_path=str(backup_path) if backup_path is not None else None,
            report_path=str(report_path),
            steps=steps,
        )
        _write_report(report, report_path)

    if status != "ok":
        raise typer.Exit(code=1)

    console.print("\n[bold green]Migration reset/reindex completed[/bold green]")
    console.print(f"Run ID: {run_id}")
    console.print(f"Report: {report_path}")
    if backup_path is not None:
        console.print(f"Backup: {backup_path}")
    return report_path
