from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from cv_rag.curation.s2_client import SemanticScholarClient
from cv_rag.curation.service import (
    CurateOptions,
    CurateThresholds,
    curate_corpus,
    load_venue_whitelist,
)
from cv_rag.shared.settings import Settings
from cv_rag.storage.sqlite import SQLiteStore


def run_curate_command(
    *,
    settings: Settings,
    console: Console,
    refresh_days: int,
    tier0_venues: Path,
    tier0_min_citations: int,
    tier0_min_cpy: float,
    tier1_min_citations: int,
    tier1_min_cpy: float,
    limit: int | None,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
    s2_client_cls: type[SemanticScholarClient] = SemanticScholarClient,
) -> None:
    sqlite_store = sqlite_store_cls(settings.sqlite_path)
    sqlite_store.create_schema()

    try:
        try:
            tier0_whitelist = load_venue_whitelist(tier0_venues)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from None

        thresholds = CurateThresholds(
            tier0_min_citations=tier0_min_citations,
            tier0_min_cpy=tier0_min_cpy,
            tier1_min_citations=tier1_min_citations,
            tier1_min_cpy=tier1_min_cpy,
        )
        options = CurateOptions(
            refresh_days=refresh_days,
            limit=limit,
            thresholds=thresholds,
        )
        s2_client = s2_client_cls(
            timeout_seconds=settings.http_timeout_seconds,
            user_agent=settings.user_agent,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=settings.arxiv_backoff_start_seconds,
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
        )

        def on_progress(done: int, total: int, arxiv_id: str, status: str) -> None:
            console.print(f"[{done}/{total}] {arxiv_id} {status}")

        console.print("[bold]Curating corpus with Semantic Scholar metadata...[/bold]")
        console.print(f"Tier-0 venue whitelist size: {len(tier0_whitelist)}")
        result = curate_corpus(
            sqlite_store=sqlite_store,
            s2_client=s2_client,
            venue_whitelist=tier0_whitelist,
            options=options,
            progress_callback=on_progress,
        )
    finally:
        sqlite_store.close()

    tier_counts = ", ".join(
        f"tier{tier}:{count}" for tier, count in sorted(result.tier_distribution.items())
    ) or "none"
    console.print("\n[bold green]Curation complete[/bold green]")
    console.print(f"Candidates: {result.total_ids}")
    console.print(f"Refreshed candidates: {result.to_refresh}")
    console.print(f"Updated: {result.updated}")
    console.print(f"Skipped: {result.skipped}")
    console.print(f"Tier distribution: {tier_counts}")
