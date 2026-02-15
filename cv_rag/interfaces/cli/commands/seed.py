from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cv_rag.seeding.awesome import seed_awesome_sources
from cv_rag.seeding.openalex import resolve_dois_openalex
from cv_rag.seeding.visionbib import seed_visionbib_sources
from cv_rag.shared.settings import Settings


def run_seed_awesome_command(
    *,
    settings: Settings,
    console: Console,
    sources: Path,
    out_dir: Path,
    tier_a_arxiv: Path,
    tier_a_dois: Path,
) -> None:
    try:
        stats = seed_awesome_sources(
            sources_path=sources,
            out_dir=out_dir,
            user_agent=settings.user_agent,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
            tier_a_arxiv_path=tier_a_arxiv,
            tier_a_dois_path=tier_a_dois,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]Awesome seeding complete[/bold green]")
    console.print(f"Repos processed: {stats.repos_processed}")
    console.print(f"Total arXiv matches: {stats.total_matches}")
    console.print(f"Unique arXiv IDs: {stats.unique_ids}")
    console.print(f"Total DOI matches: {stats.total_doi_matches}")
    console.print(f"Unique DOIs: {stats.unique_dois}")
    console.print(f"arXiv JSONL output: {stats.jsonl_path}")
    console.print(f"DOI JSONL output: {stats.doi_jsonl_path}")
    console.print(f"Tier-A arXiv (legacy): {stats.tier_a_seed_path}")
    console.print(f"Tier-A arXiv: {stats.tier_a_arxiv_path}")
    console.print(f"Tier-A DOIs: {stats.tier_a_dois_path}")

    top_repos = stats.top_repos(limit=10)
    if top_repos:
        table = Table(title="Top Repos by arXiv Match Count")
        table.add_column("Repo")
        table.add_column("Matches", justify="right")
        for repo, count in top_repos:
            table.add_row(repo, str(count))
        console.print(table)

    top_doi_repos = stats.top_doi_repos(limit=10)
    if top_doi_repos:
        table = Table(title="Top Repos by DOI Match Count")
        table.add_column("Repo")
        table.add_column("Matches", justify="right")
        for repo, count in top_doi_repos:
            table.add_row(repo, str(count))
        console.print(table)


def run_resolve_dois_command(
    *,
    settings: Settings,
    console: Console,
    dois: Path,
    out_dir: Path,
    user_agent: str | None,
    api_key_env: str,
    tier_a_urls: Path,
    tier_a_arxiv_from_openalex: Path | None,
    email: str | None,
) -> None:
    resolved_user_agent = (user_agent or settings.user_agent).strip()
    api_key = os.getenv(api_key_env)

    try:
        stats = resolve_dois_openalex(
            dois_path=dois,
            out_dir=out_dir,
            user_agent=resolved_user_agent,
            email=email,
            api_key=api_key,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
            tier_a_urls_path=tier_a_urls,
            tier_a_arxiv_path=tier_a_arxiv_from_openalex,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]OpenAlex DOI resolution complete[/bold green]")
    console.print(f"DOIs processed: {stats.dois_processed}")
    console.print(f"Resolved records: {stats.resolved_records}")
    console.print(f"Resolved OA PDF URLs: {stats.resolved_pdf_urls}")
    console.print(f"Unresolved: {stats.unresolved}")
    console.print(f"Cache hits: {stats.cache_hits}")
    console.print(f"Resolved JSONL output: {stats.jsonl_path}")
    console.print(f"Tier-A OA URL output: {stats.tier_a_urls_path}")
    if stats.tier_a_arxiv_path is not None:
        console.print(f"Recovered arXiv IDs from OpenAlex: {stats.resolved_arxiv_ids}")
        console.print(f"Tier-A arXiv-from-OpenAlex output: {stats.tier_a_arxiv_path}")
    console.print(f"Cache directory: {stats.cache_dir}")


def run_seed_visionbib_command(
    *,
    settings: Settings,
    console: Console,
    sources: Path,
    out_dir: Path,
    tier_a_dois: Path,
    tier_a_urls: Path,
    tier_a_arxiv: Path,
) -> None:
    try:
        stats = seed_visionbib_sources(
            sources_path=sources,
            out_dir=out_dir,
            user_agent=settings.user_agent,
            timeout_seconds=settings.http_timeout_seconds,
            max_retries=settings.arxiv_max_retries,
            backoff_start_seconds=max(0.5, settings.arxiv_backoff_start_seconds / 2),
            backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
            delay_seconds=0.2,
            tier_a_dois_path=tier_a_dois,
            tier_a_urls_path=tier_a_urls,
            tier_a_arxiv_path=tier_a_arxiv,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]VisionBib seeding complete[/bold green]")
    console.print(f"Pages requested: {stats.pages_requested}")
    console.print(f"Pages succeeded: {stats.pages_succeeded}")
    console.print(f"Pages failed: {stats.pages_failed}")
    console.print(f"Total DOI matches: {stats.total_doi_matches}")
    console.print(f"Total PDF URL matches: {stats.total_pdf_matches}")
    console.print(f"Total explicit arXiv matches: {stats.total_arxiv_matches}")
    console.print(f"Unique DOIs: {stats.unique_dois}")
    console.print(f"Unique PDF URLs: {stats.unique_pdf_urls}")
    console.print(f"Unique arXiv IDs: {stats.unique_arxiv_ids}")
    console.print(f"DOI JSONL output: {stats.doi_jsonl_path}")
    console.print(f"URL JSONL output: {stats.url_jsonl_path}")
    console.print(f"arXiv JSONL output: {stats.arxiv_jsonl_path}")
    console.print(f"Tier-A VisionBib DOIs: {stats.tier_a_dois_path}")
    console.print(f"Tier-A VisionBib URLs: {stats.tier_a_urls_path}")
    console.print(f"Tier-A VisionBib arXiv IDs: {stats.tier_a_arxiv_path}")

    top_pages = stats.top_pages(limit=10)
    if top_pages:
        table = Table(title="Top VisionBib Pages by Extracted Link Count")
        table.add_column("Page")
        table.add_column("Links", justify="right")
        for page, count in top_pages:
            table.add_row(page, str(count))
        console.print(table)
