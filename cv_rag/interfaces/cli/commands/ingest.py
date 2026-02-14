from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cv_rag.ingest.arxiv_client import fetch_cs_cv_papers, fetch_papers_by_ids
from cv_rag.ingest.dedupe import ExplicitIngestStats, LegacyVersionMigrationStats
from cv_rag.ingest.service import IngestService
from cv_rag.shared.settings import Settings


def print_migration_stats(console: Console, stats: LegacyVersionMigrationStats) -> None:
    console.print(
        "Legacy version migration: "
        f"found={stats.legacy_rows_found}, "
        f"migrated={stats.migrated}, "
        f"unresolved={stats.unresolved}"
    )
    if stats.unresolved > 0:
        console.print(
            "[yellow]Some legacy rows remain unresolved; exact-version dedupe may be incomplete for them.[/yellow]"
        )


def print_explicit_ingest_stats(console: Console, stats: ExplicitIngestStats) -> None:
    console.print(
        "Selection summary: "
        f"requested={stats.requested_ids}, "
        f"metadata_returned={stats.metadata_returned}, "
        f"resolved_to_version={stats.resolved_to_version}, "
        f"unresolved={stats.unresolved}, "
        f"skipped_existing={stats.skipped_existing}, "
        f"to_ingest={stats.to_ingest}"
    )


def run_ingest_command(
    *,
    settings: Settings,
    console: Console,
    limit: int,
    skip_ingested: bool,
    force_grobid: bool,
    embed_batch_size: int | None,
    fetch_latest_fn: object = fetch_cs_cv_papers,
    fetch_by_ids_fn: object = fetch_papers_by_ids,
    find_and_migrate_legacy_versions_fn: object,
    load_ingested_versions_fn: object,
    canonical_requested_ids_fn: object,
    filter_papers_by_exact_version_fn: object,
    build_explicit_ingest_stats_fn: object,
    load_arxiv_ids_from_jsonl_fn: object,
    run_ingest_fn: object,
) -> None:
    service = IngestService(
        settings,
        fetch_latest_fn=fetch_latest_fn,
        fetch_by_ids_fn=fetch_by_ids_fn,
        find_and_migrate_legacy_versions_fn=find_and_migrate_legacy_versions_fn,
        load_ingested_versions_fn=load_ingested_versions_fn,
        canonical_requested_ids_fn=canonical_requested_ids_fn,
        filter_papers_by_exact_version_fn=filter_papers_by_exact_version_fn,
        build_explicit_ingest_stats_fn=build_explicit_ingest_stats_fn,
        load_arxiv_ids_from_jsonl_fn=load_arxiv_ids_from_jsonl_fn,
    )
    selection = service.ingest_latest(limit=limit, skip_ingested=skip_ingested)

    if skip_ingested and selection.migration is not None:
        print_migration_stats(console, selection.migration)

    console.print(
        "[bold]Fetching newest cs.CV papers from arXiv "
        f"(target_new={limit}, skip_ingested={'on' if skip_ingested else 'off'})...[/bold]"
    )
    console.print(
        "Fetch summary: "
        f"requested={selection.fetch_stats.get('requested', limit)}, "
        f"selected={selection.fetch_stats.get('selected', len(selection.papers))}, "
        f"skipped={selection.fetch_stats.get('skipped', 0)}"
    )
    if selection.skipped_after_fetch > 0:
        console.print(f"Post-fetch exact-version skip: {selection.skipped_after_fetch}")

    papers = selection.papers
    if not papers:
        if skip_ingested:
            console.print(
                "[yellow]No new cs.CV papers to ingest after skipping already ingested versions.[/yellow]"
            )
            return
        console.print("[yellow]No papers returned from arXiv.[/yellow]")
        raise typer.Exit(code=1)

    run_ingest_fn(
        papers=papers,
        metadata_json_path=settings.metadata_json_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
    )


def run_ingest_ids_command(
    *,
    settings: Settings,
    console: Console,
    ids: list[str],
    skip_ingested: bool,
    force_grobid: bool,
    embed_batch_size: int | None,
    fetch_latest_fn: object = fetch_cs_cv_papers,
    fetch_by_ids_fn: object = fetch_papers_by_ids,
    find_and_migrate_legacy_versions_fn: object,
    load_ingested_versions_fn: object,
    canonical_requested_ids_fn: object,
    filter_papers_by_exact_version_fn: object,
    build_explicit_ingest_stats_fn: object,
    load_arxiv_ids_from_jsonl_fn: object,
    run_ingest_fn: object,
) -> None:
    service = IngestService(
        settings,
        fetch_latest_fn=fetch_latest_fn,
        fetch_by_ids_fn=fetch_by_ids_fn,
        find_and_migrate_legacy_versions_fn=find_and_migrate_legacy_versions_fn,
        load_ingested_versions_fn=load_ingested_versions_fn,
        canonical_requested_ids_fn=canonical_requested_ids_fn,
        filter_papers_by_exact_version_fn=filter_papers_by_exact_version_fn,
        build_explicit_ingest_stats_fn=build_explicit_ingest_stats_fn,
        load_arxiv_ids_from_jsonl_fn=load_arxiv_ids_from_jsonl_fn,
    )
    selection = service.ingest_ids(ids=ids, skip_ingested=skip_ingested)
    requested_ids = selection.requested_ids
    if not requested_ids:
        console.print("[yellow]No valid arXiv IDs provided.[/yellow]")
        raise typer.Exit(code=1)

    if skip_ingested and selection.migration is not None:
        print_migration_stats(console, selection.migration)

    console.print(
        "[bold]Fetching metadata for explicit arXiv IDs "
        f"({len(requested_ids)}, skip_ingested={'on' if skip_ingested else 'off'})...[/bold]"
    )
    print_explicit_ingest_stats(console, selection.stats)
    if skip_ingested and selection.stats.unresolved > 0:
        console.print(
            "[yellow]Some IDs could not be resolved to an exact version; "
            "duplicate prevention is best-effort for them.[/yellow]"
        )

    papers = selection.papers
    if not papers:
        if skip_ingested:
            console.print("[yellow]All requested papers are already ingested at the same version.[/yellow]")
            return
        console.print("[yellow]No valid arXiv IDs provided.[/yellow]")
        raise typer.Exit(code=1)

    metadata_path = settings.metadata_dir / "arxiv_selected_ids.json"
    run_ingest_fn(
        papers=papers,
        metadata_json_path=metadata_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
    )


def run_ingest_jsonl_command(
    *,
    settings: Settings,
    console: Console,
    source: Path,
    limit: int | None,
    skip_ingested: bool,
    force_grobid: bool,
    embed_batch_size: int | None,
    fetch_latest_fn: object = fetch_cs_cv_papers,
    fetch_by_ids_fn: object = fetch_papers_by_ids,
    find_and_migrate_legacy_versions_fn: object,
    load_ingested_versions_fn: object,
    canonical_requested_ids_fn: object,
    filter_papers_by_exact_version_fn: object,
    build_explicit_ingest_stats_fn: object,
    load_arxiv_ids_from_jsonl_fn: object,
    run_ingest_fn: object,
) -> None:
    if limit is not None and limit <= 0:
        console.print("[red]--limit must be > 0 when provided.[/red]")
        raise typer.Exit(code=1)

    service = IngestService(
        settings,
        fetch_latest_fn=fetch_latest_fn,
        fetch_by_ids_fn=fetch_by_ids_fn,
        find_and_migrate_legacy_versions_fn=find_and_migrate_legacy_versions_fn,
        load_ingested_versions_fn=load_ingested_versions_fn,
        canonical_requested_ids_fn=canonical_requested_ids_fn,
        filter_papers_by_exact_version_fn=filter_papers_by_exact_version_fn,
        build_explicit_ingest_stats_fn=build_explicit_ingest_stats_fn,
        load_arxiv_ids_from_jsonl_fn=load_arxiv_ids_from_jsonl_fn,
    )
    try:
        selection = service.ingest_jsonl(source=source, limit=limit, skip_ingested=skip_ingested)
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    requested_ids = selection.requested_ids
    if not requested_ids:
        console.print(f"[yellow]No valid arXiv IDs found in JSONL: {source}[/yellow]")
        raise typer.Exit(code=1)

    if skip_ingested and selection.migration is not None:
        print_migration_stats(console, selection.migration)

    skip_state = "on" if skip_ingested else "off"
    console.print(
        "[bold]Fetching metadata for arXiv IDs from JSONL "
        f"({len(requested_ids)} IDs, skipped={selection.skipped_records}, skip_ingested={skip_state})...[/bold]"
    )
    print_explicit_ingest_stats(console, selection.stats)
    if skip_ingested and selection.stats.unresolved > 0:
        console.print(
            "[yellow]Some IDs could not be resolved to an exact version; "
            "duplicate prevention is best-effort for them.[/yellow]"
        )

    papers = selection.papers
    if not papers:
        if skip_ingested:
            console.print("[yellow]All requested papers are already ingested at the same version.[/yellow]")
            return
        console.print("[yellow]No valid arXiv IDs provided.[/yellow]")
        raise typer.Exit(code=1)

    metadata_path = settings.metadata_dir / f"{source.stem}_selected_ids.json"
    run_ingest_fn(
        papers=papers,
        metadata_json_path=metadata_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
    )
