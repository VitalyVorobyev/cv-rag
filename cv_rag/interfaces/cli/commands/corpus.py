from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cv_rag.ingest.service import IngestService
from cv_rag.seeding.awesome import discover_awesome_references
from cv_rag.seeding.openalex import resolve_dois_openalex
from cv_rag.seeding.visionbib import discover_visionbib_references
from cv_rag.shared.settings import Settings


def _default_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_corpus_discover_awesome_command(
    *,
    settings: Settings,
    console: Console,
    sources: Path,
    run_id: str | None,
) -> None:
    use_run_id = run_id or _default_run_id()
    try:
        refs = discover_awesome_references(
            sources_path=sources,
            run_id=use_run_id,
            user_agent=settings.user_agent,
            runs_dir=settings.discovery_runs_dir,
            sqlite_path=settings.sqlite_path,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]Corpus discover (awesome) complete[/bold green]")
    console.print(f"Run ID: {use_run_id}")
    console.print(f"References discovered: {len(refs)}")
    console.print(f"Run artifact dir: {settings.discovery_runs_dir / use_run_id}")


def run_corpus_discover_visionbib_command(
    *,
    settings: Settings,
    console: Console,
    sources: Path,
    run_id: str | None,
) -> None:
    use_run_id = run_id or _default_run_id()
    try:
        refs = discover_visionbib_references(
            sources_path=sources,
            run_id=use_run_id,
            user_agent=settings.user_agent,
            runs_dir=settings.discovery_runs_dir,
            sqlite_path=settings.sqlite_path,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]Corpus discover (visionbib) complete[/bold green]")
    console.print(f"Run ID: {use_run_id}")
    console.print(f"References discovered: {len(refs)}")
    console.print(f"Run artifact dir: {settings.discovery_runs_dir / use_run_id}")


def run_corpus_resolve_openalex_command(
    *,
    settings: Settings,
    console: Console,
    dois: Path,
    out_dir: Path,
    run_id: str | None,
    email: str | None,
    api_key: str | None,
) -> None:
    use_run_id = run_id or _default_run_id()
    try:
        stats = resolve_dois_openalex(
            dois_path=dois,
            out_dir=out_dir,
            user_agent=settings.user_agent,
            email=email,
            api_key=api_key,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
            run_id=use_run_id,
            runs_dir=settings.discovery_runs_dir,
            sqlite_path=settings.sqlite_path,
            candidate_retry_days=settings.candidate_retry_days,
            candidate_max_retries=settings.candidate_max_retries,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]Corpus resolve (OpenAlex) complete[/bold green]")
    console.print(f"Run ID: {stats.run_id}")
    console.print(f"DOIs processed: {stats.dois_processed}")
    console.print(f"Resolved records: {stats.resolved_records}")
    console.print(f"Unresolved: {stats.unresolved}")
    if stats.run_artifact_path is not None:
        console.print(f"Run artifact: {stats.run_artifact_path}")


def run_corpus_queue_command(
    *,
    settings: Settings,
    console: Console,
    limit: int,
) -> None:
    service = IngestService(settings)
    candidates = service.list_ready_candidates(limit=limit)
    if not candidates:
        console.print("[yellow]No ready candidates in queue.[/yellow]")
        return

    table = Table(title=f"Ready candidates (top {len(candidates)})")
    table.add_column("Doc ID")
    table.add_column("Status")
    table.add_column("Priority", justify="right")
    table.add_column("Retries", justify="right")
    for candidate in candidates:
        table.add_row(
            candidate.doc_id,
            candidate.status,
            f"{candidate.priority_score:.3f}",
            str(candidate.retry_count),
        )
    console.print(table)


def run_corpus_ingest_command(
    *,
    settings: Settings,
    console: Console,
    limit: int,
    force_grobid: bool,
    embed_batch_size: int | None,
    cache_only: bool,
) -> None:
    service = IngestService(settings)
    candidates = service.list_ready_candidates(limit=limit)
    if not candidates:
        console.print("[yellow]No ready candidates in queue.[/yellow]")
        return

    console.print(
        f"[cyan]Starting queue ingest: selected {len(candidates)} candidate(s) (limit={limit})[/cyan]"
    )
    started_at = time.perf_counter()

    def _on_paper_progress(index: int, total: int, paper: object) -> None:
        paper_obj = paper
        # Ingest callbacks emit PaperMetadata; keep runtime-safe in case of test doubles.
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
    elapsed_seconds = time.perf_counter() - started_at
    console.print("\n[bold green]Corpus ingest from queue complete[/bold green]")
    console.print(f"Selected candidates: {stats['selected']}")
    console.print(f"Queued for ingest: {stats['queued']}")
    console.print(f"Ingested: {stats['ingested']}")
    console.print(f"Failed: {stats['failed']}")
    console.print(f"Blocked: {stats['blocked']}")
    console.print(f"Elapsed seconds: {elapsed_seconds:.1f}")
