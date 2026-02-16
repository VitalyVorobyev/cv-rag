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
from cv_rag.storage.repositories import ReferenceRecord, ResolvedReference
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


def _latest_run_artifact(
    *,
    runs_dir: Path,
    run_suffix: str,
    artifact_name: str,
) -> Path | None:
    candidates = sorted(
        [p for p in runs_dir.glob(f"*-{run_suffix}") if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for run_dir in candidates:
        artifact_path = run_dir / artifact_name
        if artifact_path.exists():
            return artifact_path
    return None


def _read_reference_records(path: Path) -> list[ReferenceRecord]:
    refs: list[ReferenceRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        refs.append(
            ReferenceRecord(
                ref_type=str(payload["ref_type"]),
                normalized_value=str(payload["normalized_value"]),
                source_kind=str(payload["source_kind"]),
                source_ref=str(payload["source_ref"]),
                discovered_at_unix=int(payload["discovered_at_unix"]),
            )
        )
    return refs


def _read_resolution_records(path: Path) -> list[ResolvedReference]:
    resolved: list[ResolvedReference] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        resolved.append(
            ResolvedReference(
                doc_id=str(payload["doc_id"]),
                arxiv_id=(
                    str(payload["arxiv_id"])
                    if isinstance(payload.get("arxiv_id"), str) and str(payload["arxiv_id"]).strip()
                    else None
                ),
                arxiv_id_with_version=(
                    str(payload["arxiv_id_with_version"])
                    if isinstance(payload.get("arxiv_id_with_version"), str)
                    and str(payload["arxiv_id_with_version"]).strip()
                    else None
                ),
                doi=(
                    str(payload["doi"])
                    if isinstance(payload.get("doi"), str) and str(payload["doi"]).strip()
                    else None
                ),
                pdf_url=(
                    str(payload["pdf_url"])
                    if isinstance(payload.get("pdf_url"), str) and str(payload["pdf_url"]).strip()
                    else None
                ),
                resolution_confidence=float(payload.get("resolution_confidence", 0.0)),
                source_kind=str(payload.get("source_kind", "openalex_resolved")),
                resolved_at_unix=(
                    int(payload["resolved_at_unix"])
                    if payload.get("resolved_at_unix") is not None
                    else None
                ),
            )
        )
    return resolved


def _restore_cached_reference_graph(
    *,
    settings: Settings,
    run_id: str,
    awesome_refs_path: Path | None,
    visionbib_refs_path: Path | None,
    openalex_resolution_path: Path | None,
    sqlite_store_cls: type[SQLiteStore],
) -> dict[str, object]:
    runs_dir = settings.discovery_runs_dir
    use_awesome_refs = awesome_refs_path or _latest_run_artifact(
        runs_dir=runs_dir,
        run_suffix="awesome",
        artifact_name="awesome_references.jsonl",
    )
    use_visionbib_refs = visionbib_refs_path or _latest_run_artifact(
        runs_dir=runs_dir,
        run_suffix="visionbib",
        artifact_name="visionbib_references.jsonl",
    )
    use_openalex_resolved = openalex_resolution_path or _latest_run_artifact(
        runs_dir=runs_dir,
        run_suffix="openalex",
        artifact_name="openalex_resolution.jsonl",
    )

    refs: list[ReferenceRecord] = []
    resolved: list[ResolvedReference] = []
    if use_awesome_refs is not None:
        refs.extend(_read_reference_records(use_awesome_refs))
    if use_visionbib_refs is not None:
        refs.extend(_read_reference_records(use_visionbib_refs))
    if use_openalex_resolved is not None:
        resolved.extend(_read_resolution_records(use_openalex_resolved))

    if not refs and not resolved:
        raise FileNotFoundError(
            "Cache-only migration could not find any run artifacts. "
            "Provide --cache-awesome-refs / --cache-visionbib-refs / --cache-openalex-resolution "
            "or ensure data/curation/runs has prior artifacts."
        )

    sqlite_store = sqlite_store_cls(settings.sqlite_path)
    try:
        sqlite_store.create_schema()
        sqlite_store.upsert_reference_graph(
            refs=refs,
            resolved=resolved,
            run_id=run_id,
            candidate_retry_days=settings.candidate_retry_days,
            candidate_max_retries=settings.candidate_max_retries,
        )
    finally:
        sqlite_store.close()

    return {
        "refs_loaded": len(refs),
        "resolved_loaded": len(resolved),
        "awesome_refs_path": str(use_awesome_refs) if use_awesome_refs else None,
        "visionbib_refs_path": str(use_visionbib_refs) if use_visionbib_refs else None,
        "openalex_resolution_path": str(use_openalex_resolved) if use_openalex_resolved else None,
    }


def _preflight_checks(
    *,
    settings: Settings,
    awesome_sources: Path,
    visionbib_sources: Path,
    dois: Path,
    qdrant_store_cls: type[QdrantStore],
    cache_only: bool = False,
    awesome_refs_path: Path | None = None,
    visionbib_refs_path: Path | None = None,
    openalex_resolution_path: Path | None = None,
) -> dict[str, object]:
    missing_inputs: list[str] = []
    if cache_only:
        for path in (awesome_refs_path, visionbib_refs_path, openalex_resolution_path):
            if path is not None and not path.exists():
                missing_inputs.append(str(path))
        if not any((awesome_refs_path, visionbib_refs_path, openalex_resolution_path)):
            runs_dir = settings.discovery_runs_dir
            has_any = any(runs_dir.glob("*-awesome")) or any(runs_dir.glob("*-visionbib")) or any(
                runs_dir.glob("*-openalex")
            )
            if not has_any:
                missing_inputs.append(f"{runs_dir} (no prior run artifacts found)")
    else:
        for path in (awesome_sources, visionbib_sources, dois):
            if not path.exists():
                missing_inputs.append(str(path))
    if missing_inputs:
        raise FileNotFoundError("Preflight failed; missing required inputs: " + ", ".join(missing_inputs))

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
    cache_only: bool,
    max_ingest_loops: int,
    ingest_service_cls: type[IngestService],
    console: Console | None = None,
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
            if console is not None:
                console.print(
                    f"[green]Ingest queue drained after {max(loops - 1, 0)} loop(s).[/green]"
                )
            break

        if console is not None:
            console.print(
                f"[cyan]Ingest loop {loops}: processing {len(candidates)} candidate(s) "
                f"(limit={ingest_batch_size})[/cyan]"
            )

        def _on_paper_progress(index: int, total: int, paper: object) -> None:
            if console is None:
                return
            paper_obj = paper
            paper_id = getattr(paper_obj, "arxiv_id_with_version", None) or getattr(
                paper_obj, "arxiv_id", None
            )
            if not paper_id:
                resolver = getattr(paper_obj, "resolved_doc_id", None)
                if callable(resolver):
                    paper_id = str(resolver())
            if not paper_id:
                paper_id = "<unknown>"
            console.print(f"  [cyan][{index}/{total}] ingesting {paper_id}[/cyan]")

        stats = service.ingest_candidates(
            candidates=candidates,
            force_grobid=force_grobid,
            embed_batch_size=embed_batch_size,
            cache_only=cache_only,
            on_paper_progress=_on_paper_progress,
        )
        selected_total += int(stats.get("selected", 0))
        queued_total += int(stats.get("queued", 0))
        ingested_total += int(stats.get("ingested", 0))
        failed_total += int(stats.get("failed", 0))
        blocked_total += int(stats.get("blocked", 0))
        if console is not None:
            console.print(
                f"[green]Loop {loops} done:[/green] queued={int(stats.get('queued', 0))}, "
                f"ingested={int(stats.get('ingested', 0))}, failed={int(stats.get('failed', 0))}, "
                f"blocked={int(stats.get('blocked', 0))}"
            )
            console.print(
                "         cumulative: "
                f"queued={queued_total}, ingested={ingested_total}, "
                f"failed={failed_total}, blocked={blocked_total}"
            )

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
    cache_only: bool = False,
    awesome_sources: Path,
    visionbib_sources: Path,
    dois: Path,
    openalex_out_dir: Path,
    cache_awesome_refs: Path | None = None,
    cache_visionbib_refs: Path | None = None,
    cache_openalex_resolution: Path | None = None,
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
        step_number = len(steps) + 1
        console.print(f"\n[bold cyan]Step {step_number}: {name}[/bold cyan]")
        t0 = time.perf_counter()
        try:
            details = fn()  # type: ignore[operator]
            if details is None:
                details = {}
            details_dict = dict(details)
            duration = round(time.perf_counter() - t0, 3)
            steps.append(
                MigrationStep(
                    name=name,
                    status="ok",
                    duration_seconds=duration,
                    details=details_dict,
                )
            )
            console.print(f"[green]Step {name} completed in {duration:.3f}s[/green]")
            return details_dict
        except Exception as exc:
            duration = round(time.perf_counter() - t0, 3)
            steps.append(
                MigrationStep(
                    name=name,
                    status="fail",
                    duration_seconds=duration,
                    details={"error": str(exc)},
                )
            )
            console.print(f"[red]Step {name} failed after {duration:.3f}s: {exc}[/red]")
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
                cache_only=cache_only,
                awesome_refs_path=cache_awesome_refs,
                visionbib_refs_path=cache_visionbib_refs,
                openalex_resolution_path=cache_openalex_resolution,
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

        if cache_only:
            _run_step(
                "restore_cached_references",
                lambda: _restore_cached_reference_graph(
                    settings=settings,
                    run_id=f"{run_id}-cache-restore",
                    awesome_refs_path=cache_awesome_refs,
                    visionbib_refs_path=cache_visionbib_refs,
                    openalex_resolution_path=cache_openalex_resolution,
                    sqlite_store_cls=sqlite_store_cls,
                ),
            )
        else:
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
                cache_only=cache_only,
                max_ingest_loops=max_ingest_loops,
                ingest_service_cls=ingest_service_cls,
                console=console,
            ),
        )

        if skip_curate or cache_only:
            reason = "cache_only_mode" if cache_only and not skip_curate else "skip_flag"
            _run_step("curate", lambda: {"skipped": True, "reason": reason})
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
