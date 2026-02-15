from __future__ import annotations

import logging
import os
from pathlib import Path

import typer
from rich.console import Console

from cv_rag.answer.mlx_runner import mlx_generate
from cv_rag.answer.service import AnswerService
from cv_rag.curation.s2_client import SemanticScholarClient
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.ingest.arxiv_client import (
    PaperMetadata,
    fetch_cs_cv_papers,
    fetch_papers_by_ids,
)
from cv_rag.ingest.dedupe import (
    ExplicitIngestStats as _ExplicitIngestStats,
)
from cv_rag.ingest.dedupe import (
    LegacyVersionMigrationStats as _LegacyVersionMigrationStats,
)
from cv_rag.ingest.dedupe import (
    build_explicit_ingest_stats as _build_explicit_ingest_stats,
)
from cv_rag.ingest.dedupe import (
    canonical_requested_ids as _canonical_requested_ids,
)
from cv_rag.ingest.dedupe import (
    filter_papers_by_exact_version as _filter_papers_by_exact_version,
)
from cv_rag.ingest.dedupe import (
    find_and_migrate_legacy_versions as _find_and_migrate_legacy_versions,
)
from cv_rag.ingest.dedupe import (
    load_arxiv_ids_from_jsonl as _load_arxiv_ids_from_jsonl,
)
from cv_rag.ingest.dedupe import (
    load_ingested_versions as _load_ingested_versions,
)
from cv_rag.ingest.pdf_pipeline import IngestPipeline
from cv_rag.interfaces.cli.commands.answer import run_answer_command
from cv_rag.interfaces.cli.commands.corpus import (
    run_corpus_discover_awesome_command,
    run_corpus_discover_visionbib_command,
    run_corpus_ingest_command,
    run_corpus_queue_command,
    run_corpus_resolve_openalex_command,
)
from cv_rag.interfaces.cli.commands.curate import run_curate_command
from cv_rag.interfaces.cli.commands.doctor import run_doctor_command
from cv_rag.interfaces.cli.commands.eval import load_eval_cases, run_eval_command
from cv_rag.interfaces.cli.commands.ingest import (
    run_ingest_command,
    run_ingest_ids_command,
    run_ingest_jsonl_command,
)
from cv_rag.interfaces.cli.commands.migrate import run_migrate_reset_reindex_command
from cv_rag.interfaces.cli.commands.query import run_query_command
from cv_rag.interfaces.cli.commands.serve import run_serve_command
from cv_rag.interfaces.cli.commands.stats import run_stats_command
from cv_rag.retrieval.hybrid import (
    HybridRetriever,
)
from cv_rag.seeding.awesome import (
    DEFAULT_TIER_A_ARXIV_PATH,
    DEFAULT_TIER_A_DOIS_PATH,
)
from cv_rag.seeding.openalex import (
    DEFAULT_TIER_A_OPENALEX_URLS_PATH,
)
from cv_rag.seeding.visionbib import (
    DEFAULT_TIER_A_VISIONBIB_ARXIV_PATH,
    DEFAULT_TIER_A_VISIONBIB_DOIS_PATH,
    DEFAULT_TIER_A_VISIONBIB_URLS_PATH,
    DEFAULT_VISIONBIB_OUT_DIR,
)
from cv_rag.shared.settings import get_settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local CV papers RAG MVP")
seed_app = typer.Typer(help="Seed curation files from external sources.")
corpus_app = typer.Typer(help="Multi-source corpus growth workflows.")
migrate_app = typer.Typer(help="Migration and corpus rebuild commands.")
app.add_typer(seed_app, name="seed")
app.add_typer(corpus_app, name="corpus")
app.add_typer(migrate_app, name="migrate")
console = Console()

# Backward-compatible symbol used by tests/tooling.
_load_eval_cases = load_eval_cases
LegacyVersionMigrationStats = _LegacyVersionMigrationStats
ExplicitIngestStats = _ExplicitIngestStats


@app.callback()
def _main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")


def _legacy_command_removed(*, command: str) -> None:
    console.print(f"[red]Command removed: {command}[/red]")
    console.print("[yellow]Use the corpus workflow commands instead:[/yellow]")
    console.print("  cv-rag corpus discover-awesome --sources <path>")
    console.print("  cv-rag corpus discover-visionbib --sources <path>")
    console.print("  cv-rag corpus resolve-openalex --dois <path> --out-dir <path>")
    console.print("  cv-rag corpus queue --limit <n>")
    console.print("  cv-rag corpus ingest --limit <n>")
    raise typer.Exit(code=1)


def _run_ingest(
    papers: list[PaperMetadata],
    metadata_json_path: Path,
    force_grobid: bool,
    embed_batch_size: int | None,
) -> None:
    """Thin CLI wrapper around IngestPipeline with console output."""
    settings = get_settings()

    if not papers:
        console.print("[yellow]No papers to ingest.[/yellow]")
        raise typer.Exit(code=1)

    def on_progress(idx: int, total: int, paper: PaperMetadata) -> None:
        console.print(f"[{idx}/{total}] {paper.arxiv_id_with_version} - {paper.title}")

    pipeline = IngestPipeline(settings)
    result = pipeline.run(
        papers=papers,
        metadata_json_path=metadata_json_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        on_progress=on_progress,
    )

    for failure in result.failed_papers:
        console.print(f"[red]  Failed: {failure}[/red]")

    console.print("\n[bold green]Ingest complete[/bold green]")
    console.print(f"Papers processed: {result.papers_processed}")
    console.print(f"Total chunks upserted: {result.total_chunks}")
    console.print(f"SQLite index: {settings.sqlite_path}")
    console.print(f"Qdrant collection: {settings.qdrant_collection}")


@app.command()
def ingest(
    limit: int = typer.Option(
        None,
        "--limit",
        "-n",
        help="Number of newest cs.CV papers to ingest.",
    ),
    skip_ingested: bool = typer.Option(
        True,
        "--skip-ingested/--no-skip-ingested",
        help="Skip papers whose exact arXiv version is already in the local SQLite index.",
    ),
    force_grobid: bool = typer.Option(
        False,
        help="Re-run GROBID parsing even when TEI already exists.",
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedding batch size override.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_ingest_command(
        settings=settings,
        console=console,
        limit=limit or settings.default_arxiv_limit,
        skip_ingested=skip_ingested,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        fetch_latest_fn=fetch_cs_cv_papers,
        fetch_by_ids_fn=fetch_papers_by_ids,
        find_and_migrate_legacy_versions_fn=_find_and_migrate_legacy_versions,
        load_ingested_versions_fn=_load_ingested_versions,
        canonical_requested_ids_fn=_canonical_requested_ids,
        filter_papers_by_exact_version_fn=_filter_papers_by_exact_version,
        build_explicit_ingest_stats_fn=_build_explicit_ingest_stats,
        load_arxiv_ids_from_jsonl_fn=_load_arxiv_ids_from_jsonl,
        run_ingest_fn=_run_ingest,
    )


@app.command("ingest-ids")
def ingest_ids(
    ids: list[str] = typer.Argument(
        ...,
        help="One or more arXiv IDs to ingest (example: 2104.00680 1911.11763).",
    ),
    skip_ingested: bool = typer.Option(
        True,
        "--skip-ingested/--no-skip-ingested",
        help="Skip papers whose exact arXiv version is already in the local SQLite index.",
    ),
    force_grobid: bool = typer.Option(
        False,
        help="Re-run GROBID parsing even when TEI already exists.",
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedding batch size override.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_ingest_ids_command(
        settings=settings,
        console=console,
        ids=ids,
        skip_ingested=skip_ingested,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        fetch_latest_fn=fetch_cs_cv_papers,
        fetch_by_ids_fn=fetch_papers_by_ids,
        find_and_migrate_legacy_versions_fn=_find_and_migrate_legacy_versions,
        load_ingested_versions_fn=_load_ingested_versions,
        canonical_requested_ids_fn=_canonical_requested_ids,
        filter_papers_by_exact_version_fn=_filter_papers_by_exact_version,
        build_explicit_ingest_stats_fn=_build_explicit_ingest_stats,
        load_arxiv_ids_from_jsonl_fn=_load_arxiv_ids_from_jsonl,
        run_ingest_fn=_run_ingest,
    )


@app.command("ingest-jsonl")
def ingest_jsonl(
    source: Path = typer.Option(
        ...,
        "--source",
        help="Path to JSONL file containing arXiv IDs (e.g., awesome_seed.jsonl).",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Optional maximum number of IDs to ingest from the JSONL file.",
    ),
    skip_ingested: bool = typer.Option(
        True,
        "--skip-ingested/--no-skip-ingested",
        help="Skip papers whose exact arXiv version is already in the local SQLite index.",
    ),
    force_grobid: bool = typer.Option(
        False,
        help="Re-run GROBID parsing even when TEI already exists.",
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedding batch size override.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_ingest_jsonl_command(
        settings=settings,
        console=console,
        source=source,
        limit=limit,
        skip_ingested=skip_ingested,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        fetch_latest_fn=fetch_cs_cv_papers,
        fetch_by_ids_fn=fetch_papers_by_ids,
        find_and_migrate_legacy_versions_fn=_find_and_migrate_legacy_versions,
        load_ingested_versions_fn=_load_ingested_versions,
        canonical_requested_ids_fn=_canonical_requested_ids,
        filter_papers_by_exact_version_fn=_filter_papers_by_exact_version,
        build_explicit_ingest_stats_fn=_build_explicit_ingest_stats,
        load_arxiv_ids_from_jsonl_fn=_load_arxiv_ids_from_jsonl,
        run_ingest_fn=_run_ingest,
    )


@app.command()
def curate(
    refresh_days: int = typer.Option(
        30,
        "--refresh-days",
        help="Refresh metrics older than this many days.",
    ),
    tier0_venues: Path = typer.Option(
        Path("data/venues_tier0.txt"),
        "--tier0-venues",
        help="Path to tier-0 venue list (txt or yaml).",
    ),
    tier0_min_citations: int = typer.Option(
        200,
        "--tier0-min-citations",
        help="Tier-0 minimum citation count threshold.",
    ),
    tier0_min_cpy: float = typer.Option(
        30.0,
        "--tier0-min-cpy",
        help="Tier-0 minimum citations-per-year threshold.",
    ),
    tier1_min_citations: int = typer.Option(
        20,
        "--tier1-min-citations",
        help="Tier-1 minimum citation count threshold.",
    ),
    tier1_min_cpy: float = typer.Option(
        3.0,
        "--tier1-min-cpy",
        help="Tier-1 minimum citations-per-year threshold.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        help="Only enrich first N arXiv IDs from the local paper index.",
    ),
) -> None:
    settings = get_settings()
    run_curate_command(
        settings=settings,
        console=console,
        refresh_days=refresh_days,
        tier0_venues=tier0_venues,
        tier0_min_citations=tier0_min_citations,
        tier0_min_cpy=tier0_min_cpy,
        tier1_min_citations=tier1_min_citations,
        tier1_min_cpy=tier1_min_cpy,
        limit=limit,
        sqlite_store_cls=SQLiteStore,
        s2_client_cls=SemanticScholarClient,
    )


@seed_app.command("awesome")
def seed_awesome(
    sources: Path = typer.Option(
        ...,
        "--sources",
        help="Path to GitHub repo list (owner/repo or full GitHub URL).",
    ),
    out_dir: Path = typer.Option(
        Path("data/curation/seeds"),
        "--out-dir",
        help="Output directory for awesome_seed.jsonl and awesome_seed_doi.jsonl.",
    ),
    tier_a_arxiv: Path = typer.Option(
        DEFAULT_TIER_A_ARXIV_PATH,
        "--tierA-arxiv",
        help="Path for unique arXiv base-id seed file.",
    ),
    tier_a_dois: Path = typer.Option(
        DEFAULT_TIER_A_DOIS_PATH,
        "--tierA-dois",
        help="Path for unique DOI seed file.",
    ),
) -> None:
    _ = (sources, out_dir, tier_a_arxiv, tier_a_dois)
    _legacy_command_removed(command="cv-rag seed awesome")


@seed_app.command("visionbib")
def seed_visionbib(
    sources: Path = typer.Option(
        ...,
        "--sources",
        help="Path to VisionBib source file (prefix URL + page ranges).",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_VISIONBIB_OUT_DIR,
        "--out-dir",
        help="Output directory for VisionBib JSONL seed files.",
    ),
    tier_a_dois: Path = typer.Option(
        DEFAULT_TIER_A_VISIONBIB_DOIS_PATH,
        "--tierA-dois",
        help="Path for VisionBib DOI seed file.",
    ),
    tier_a_urls: Path = typer.Option(
        DEFAULT_TIER_A_VISIONBIB_URLS_PATH,
        "--tierA-urls",
        help="Path for VisionBib direct PDF URL seed file.",
    ),
    tier_a_arxiv: Path = typer.Option(
        DEFAULT_TIER_A_VISIONBIB_ARXIV_PATH,
        "--tierA-arxiv",
        help="Path for explicit VisionBib arXiv seed file.",
    ),
) -> None:
    _ = (sources, out_dir, tier_a_dois, tier_a_urls, tier_a_arxiv)
    _legacy_command_removed(command="cv-rag seed visionbib")


@app.command("seed-awesome")
def seed_awesome_root(
    sources: Path = typer.Option(
        ...,
        "--sources",
        help="Path to GitHub repo list (owner/repo or full GitHub URL).",
    ),
    out_dir: Path = typer.Option(
        Path("data/curation/seeds"),
        "--out-dir",
        help="Output directory for awesome_seed.jsonl and awesome_seed_doi.jsonl.",
    ),
    tier_a_arxiv: Path = typer.Option(
        DEFAULT_TIER_A_ARXIV_PATH,
        "--tierA-arxiv",
        help="Path for unique arXiv base-id seed file.",
    ),
    tier_a_dois: Path = typer.Option(
        DEFAULT_TIER_A_DOIS_PATH,
        "--tierA-dois",
        help="Path for unique DOI seed file.",
    ),
) -> None:
    _ = (sources, out_dir, tier_a_arxiv, tier_a_dois)
    _legacy_command_removed(command="cv-rag seed-awesome")


@app.command("resolve-dois")
def resolve_dois(
    dois: Path = typer.Option(
        DEFAULT_TIER_A_DOIS_PATH,
        "--dois",
        help="Path to DOI seed file (one DOI per line).",
    ),
    out_dir: Path = typer.Option(
        Path("data/curation"),
        "--out-dir",
        help="Output directory for openalex_resolved.jsonl and cache.",
    ),
    user_agent: str | None = typer.Option(
        None,
        "--user-agent",
        help="User-Agent string for OpenAlex requests. Defaults to CV_RAG_USER_AGENT.",
    ),
    api_key_env: str = typer.Option(
        "OPENALEX_API_KEY",
        "--api-key-env",
        help="Environment variable name containing the OpenAlex API key.",
    ),
    tier_a_urls: Path = typer.Option(
        DEFAULT_TIER_A_OPENALEX_URLS_PATH,
        "--tierA-urls",
        help="Path to write resolved OpenAlex OA PDF URLs.",
    ),
    tier_a_arxiv_from_openalex: Path | None = typer.Option(
        None,
        "--tierA-arxiv-from-openalex",
        help="Optional path to write arXiv IDs recovered from OpenAlex DOI metadata.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Optional contact email appended to User-Agent as mailto.",
    ),
) -> None:
    _ = (dois, out_dir, user_agent, api_key_env, tier_a_urls, tier_a_arxiv_from_openalex, email)
    _legacy_command_removed(command="cv-rag resolve-dois")


@migrate_app.command("reset-reindex")
def migrate_reset_reindex(
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Acknowledge destructive local reset/reindex behavior.",
    ),
    backup_dir: Path | None = typer.Option(
        None,
        "--backup-dir",
        help="Optional path to store a SQLite backup before reset.",
    ),
    skip_curate: bool = typer.Option(
        False,
        "--skip-curate",
        help="Skip final curation step after corpus rebuild.",
    ),
    cache_only: bool = typer.Option(
        False,
        "--cache-only",
        help=(
            "Do not fetch remote discovery/PDF data; restore references from cached run "
            "artifacts and ingest local PDF cache only."
        ),
    ),
    awesome_sources: Path = typer.Option(
        Path("data/curation/awesome_sources.txt"),
        "--awesome-sources",
        help="Path to awesome source repo list.",
    ),
    visionbib_sources: Path = typer.Option(
        Path("data/curation/visionbib_sources.txt"),
        "--visionbib-sources",
        help="Path to VisionBib source list.",
    ),
    dois: Path = typer.Option(
        DEFAULT_TIER_A_DOIS_PATH,
        "--dois",
        help="Path to DOI list for OpenAlex resolution.",
    ),
    openalex_out_dir: Path = typer.Option(
        Path("data/curation"),
        "--openalex-out-dir",
        help="Output directory for OpenAlex artifacts.",
    ),
    cache_awesome_refs: Path | None = typer.Option(
        None,
        "--cache-awesome-refs",
        help="Optional path to cached awesome references JSONL.",
    ),
    cache_visionbib_refs: Path | None = typer.Option(
        None,
        "--cache-visionbib-refs",
        help="Optional path to cached VisionBib references JSONL.",
    ),
    cache_openalex_resolution: Path | None = typer.Option(
        None,
        "--cache-openalex-resolution",
        help="Optional path to cached OpenAlex resolution JSONL.",
    ),
    ingest_batch_size: int = typer.Option(
        10,
        "--ingest-batch-size",
        help="Ready-candidate batch size for each ingest loop iteration.",
    ),
    force_grobid: bool = typer.Option(
        False,
        "--force-grobid",
        help="Force GROBID re-parse during queue ingest.",
    ),
    embed_batch_size: int = typer.Option(
        None,
        "--embed-batch-size",
        help="Embedding batch size override for queue ingest.",
    ),
    max_ingest_loops: int = typer.Option(
        500,
        "--max-ingest-loops",
        help="Safety cap on ingest loop iterations.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_migrate_reset_reindex_command(
        settings=settings,
        console=console,
        yes=yes,
        backup_dir=backup_dir,
        skip_curate=skip_curate,
        cache_only=cache_only,
        awesome_sources=awesome_sources,
        visionbib_sources=visionbib_sources,
        dois=dois,
        openalex_out_dir=openalex_out_dir,
        cache_awesome_refs=cache_awesome_refs,
        cache_visionbib_refs=cache_visionbib_refs,
        cache_openalex_resolution=cache_openalex_resolution,
        ingest_batch_size=ingest_batch_size,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        max_ingest_loops=max_ingest_loops,
    )


@corpus_app.command("discover-awesome")
def corpus_discover_awesome(
    sources: Path = typer.Option(
        ...,
        "--sources",
        help="Path to GitHub repo list (owner/repo or full GitHub URL).",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID. Defaults to UTC timestamp.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_corpus_discover_awesome_command(
        settings=settings,
        console=console,
        sources=sources,
        run_id=run_id,
    )


@corpus_app.command("discover-visionbib")
def corpus_discover_visionbib(
    sources: Path = typer.Option(
        ...,
        "--sources",
        help="Path to VisionBib source file (prefix URL + page ranges).",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID. Defaults to UTC timestamp.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_corpus_discover_visionbib_command(
        settings=settings,
        console=console,
        sources=sources,
        run_id=run_id,
    )


@corpus_app.command("resolve-openalex")
def corpus_resolve_openalex(
    dois: Path = typer.Option(
        DEFAULT_TIER_A_DOIS_PATH,
        "--dois",
        help="Path to DOI seed file (one DOI per line).",
    ),
    out_dir: Path = typer.Option(
        Path("data/curation"),
        "--out-dir",
        help="Output directory for OpenAlex resolution outputs.",
    ),
    run_id: str | None = typer.Option(
        None,
        "--run-id",
        help="Optional run ID. Defaults to UTC timestamp.",
    ),
    api_key_env: str = typer.Option(
        "OPENALEX_API_KEY",
        "--api-key-env",
        help="Environment variable name containing the OpenAlex API key.",
    ),
    email: str | None = typer.Option(
        None,
        "--email",
        help="Optional contact email appended to User-Agent as mailto.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_corpus_resolve_openalex_command(
        settings=settings,
        console=console,
        dois=dois,
        out_dir=out_dir,
        run_id=run_id,
        email=email,
        api_key=os.getenv(api_key_env),
    )


@corpus_app.command("queue")
def corpus_queue(
    limit: int = typer.Option(
        25,
        "--limit",
        "-n",
        help="Number of ready candidates to display.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_corpus_queue_command(settings=settings, console=console, limit=limit)


@corpus_app.command("ingest")
def corpus_ingest(
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Number of ready candidates to ingest.",
    ),
    force_grobid: bool = typer.Option(
        False,
        help="Re-run GROBID parsing even when TEI already exists.",
    ),
    embed_batch_size: int = typer.Option(
        None,
        help="Embedding batch size override.",
    ),
    cache_only: bool = typer.Option(
        False,
        "--cache-only",
        help="Do not download missing PDFs; ingest only documents already present in local cache.",
    ),
) -> None:
    settings = get_settings()
    settings.ensure_directories()
    run_corpus_ingest_command(
        settings=settings,
        console=console,
        limit=limit,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
        cache_only=cache_only,
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask against local index."),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Final number of merged chunks to keep."),
    vector_k: int = typer.Option(12, "--vector-k", help="Top-k vector hits from Qdrant."),
    keyword_k: int = typer.Option(12, "--keyword-k", help="Top-k keyword hits from SQLite FTS."),
    max_per_doc: int = typer.Option(
        4,
        "--max-per-doc",
        help="Maximum number of chunks to keep per paper.",
    ),
    section_boost: float = typer.Option(
        0.0,
        "--section-boost",
        help="Additive score boost for method/training-oriented section/title matches.",
    ),
) -> None:
    run_query_command(
        settings=get_settings(),
        console=console,
        question=question,
        top_k=top_k,
        vector_k=vector_k,
        keyword_k=keyword_k,
        max_per_doc=max_per_doc,
        section_boost=section_boost,
        qdrant_store_cls=QdrantStore,
        sqlite_store_cls=SQLiteStore,
        embed_client_cls=OllamaEmbedClient,
        retriever_cls=HybridRetriever,
    )


@app.command()
def answer(
    question: str = typer.Argument(..., help="Question to answer from local index."),
    k: int | None = typer.Option(
        None,
        "--k",
        help="Override number of sources after routing (default depends on selected mode).",
    ),
    model: str = typer.Option(..., "--model", help="MLX model ID, local path, or HF repo."),
    max_tokens: int = typer.Option(600, "--max-tokens", help="Maximum tokens to generate."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    top_p: float = typer.Option(0.9, "--top-p", help="Sampling top-p."),
    seed: int | None = typer.Option(None, "--seed", help="Optional random seed."),
    max_per_doc: int | None = typer.Option(
        None,
        "--max-per-doc",
        help="Override per-paper chunk cap after routing (default depends on selected mode).",
    ),
    section_boost: float = typer.Option(
        0.05,
        "--section-boost",
        help="Additive score boost for prioritized section/title matches; set 0 to disable.",
    ),
    show_sources: bool = typer.Option(
        False,
        "--show-sources",
        help="Print retrieved sources before generating the final answer.",
    ),
    no_refuse: bool = typer.Option(
        False,
        "--no-refuse",
        help="If citation validation still fails after repair, print the draft with a warning instead of exiting.",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Answer mode: auto|explain|compare|survey|implement|evidence|decision (single alias supported).",
    ),
    router_model: str | None = typer.Option(
        None,
        "--router-model",
        help="Optional model for routing. Defaults to --model.",
    ),
    router_strategy: str = typer.Option(
        "hybrid",
        "--router-strategy",
        help="Routing strategy: rules|llm|hybrid.",
    ),
    router_top_k: int = typer.Option(
        12,
        "--router-top-k",
        help="Cheap retrieval top-k used for mode routing.",
    ),
) -> None:
    settings = get_settings()
    run_answer_command(
        settings=settings,
        console=console,
        question=question,
        model=model,
        mode=mode,
        router_model=router_model,
        router_strategy=router_strategy,
        router_top_k=router_top_k,
        k=k,
        max_per_doc=max_per_doc,
        section_boost=section_boost,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        show_sources=show_sources,
        no_refuse=no_refuse,
        mlx_generate_fn=mlx_generate,
        qdrant_store_cls=QdrantStore,
        sqlite_store_cls=SQLiteStore,
        embed_client_cls=OllamaEmbedClient,
        retriever_cls=HybridRetriever,
        answer_service_cls=AnswerService,
    )


@app.command()
def eval(
    suite: Path = typer.Option(
        Path("eval/questions.yaml"),
        "--suite",
        help="Path to YAML eval suite.",
    ),
    model: str = typer.Option(
        "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "--model",
        help="MLX model ID, local path, or HF repo.",
    ),
    k: int = typer.Option(12, "--k", help="Number of sources to include after retrieval merge."),
    max_per_doc: int = typer.Option(
        4,
        "--max-per-doc",
        help="Maximum number of chunks to keep per paper.",
    ),
    section_boost: float = typer.Option(
        0.05,
        "--section-boost",
        help="Additive score boost for prioritized section/title matches; set 0 to disable.",
    ),
    max_tokens: int = typer.Option(600, "--max-tokens", help="Maximum tokens to generate per case."),
) -> None:
    run_eval_command(
        settings=get_settings(),
        console=console,
        suite=suite,
        model=model,
        k=k,
        max_per_doc=max_per_doc,
        section_boost=section_boost,
        max_tokens=max_tokens,
        mlx_generate_fn=mlx_generate,
        qdrant_store_cls=QdrantStore,
        sqlite_store_cls=SQLiteStore,
        embed_client_cls=OllamaEmbedClient,
        retriever_cls=HybridRetriever,
    )


@app.command()
def stats(
    top_venues: int = typer.Option(
        10,
        "--top-venues",
        help="Show top-N venues from paper_metrics (0 to disable).",
    )
) -> None:
    run_stats_command(
        settings=get_settings(),
        console=console,
        top_venues=top_venues,
        sqlite_store_cls=SQLiteStore,
    )


@app.command()
def doctor(
    qdrant_test_point: bool = typer.Option(
        False,
        help="Insert one test point into Qdrant.",
    )
) -> None:
    run_doctor_command(
        settings=get_settings(),
        console=console,
        qdrant_test_point=qdrant_test_point,
        qdrant_store_cls=QdrantStore,
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
) -> None:
    """Start the web UI server."""
    run_serve_command(console=console, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
