from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx
import typer
import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table

from cv_rag.answer import (
    build_query_prompt,
    build_strict_answer_prompt,
    enforce_cross_doc_support,
    retrieve_for_answer,
    validate_answer_citations,
)
from cv_rag.arxiv_sync import (
    PaperMetadata,
    fetch_cs_cv_papers,
    fetch_papers_by_ids,
)
from cv_rag.config import get_settings
from cv_rag.curate import CurateOptions, CurateThresholds, curate_corpus, load_venue_whitelist
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.exceptions import GenerationError
from cv_rag.ingest import IngestPipeline
from cv_rag.llm import mlx_generate
from cv_rag.prompts_answer import (
    build_prompt as build_mode_answer_prompt,
)
from cv_rag.prompts_answer import (
    build_repair_prompt as build_mode_repair_prompt,
)
from cv_rag.qdrant_store import QdrantStore, VectorPoint
from cv_rag.retrieve import (
    HybridRetriever,
    RetrievedChunk,
    extract_entity_like_tokens,
    format_citation,
)
from cv_rag.routing import (
    AnswerMode,
    RouteDecision,
    make_route_decision,
    mode_from_value,
)
from cv_rag.routing import (
    route as route_answer_mode,
)
from cv_rag.s2_client import SemanticScholarClient
from cv_rag.seed_awesome import seed_awesome_sources
from cv_rag.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local CV papers RAG MVP")
seed_app = typer.Typer(help="Seed curation files from external sources.")
app.add_typer(seed_app, name="seed")
console = Console()


@app.callback()
def _main_callback(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")


class EvalCase(BaseModel):
    question: str
    must_include_arxiv_ids: list[str] = Field(default_factory=list)
    must_include_tokens: list[str] = Field(default_factory=list)
    min_sources: int = 6


@dataclass(slots=True)
class EvalCaseResult:
    index: int
    question: str
    status: bool
    retrieved_sources: int
    note: str




def _load_eval_cases(suite_path: Path) -> list[EvalCase]:
    try:
        raw_data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {suite_path}: {exc}") from exc

    if raw_data is None:
        return []
    if isinstance(raw_data, dict) and "cases" in raw_data:
        raw_data = raw_data["cases"]
    if not isinstance(raw_data, list):
        raise ValueError("Eval suite root must be a list of cases.")

    cases: list[EvalCase] = []
    for idx, entry in enumerate(raw_data, start=1):
        try:
            cases.append(EvalCase.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"Invalid eval case #{idx}: {exc}") from exc
    return cases


def _chunk_matches_eval_constraints(chunk: RetrievedChunk, case: EvalCase) -> bool:
    if case.must_include_arxiv_ids and chunk.arxiv_id not in set(case.must_include_arxiv_ids):
        return False
    if case.must_include_tokens:
        haystack = f"{chunk.title}\n{chunk.section_title}\n{chunk.text}".casefold()
        if not all(token.casefold() in haystack for token in case.must_include_tokens):
            return False
    return True


def _has_eval_constraint_match(chunks: list[RetrievedChunk], case: EvalCase) -> bool:
    if not case.must_include_arxiv_ids and not case.must_include_tokens:
        return True
    return any(_chunk_matches_eval_constraints(chunk, case) for chunk in chunks)


def _mlx_generate_or_exit(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> str:
    """Thin CLI wrapper: calls mlx_generate, converts GenerationError to typer.Exit."""
    try:
        return mlx_generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
    except GenerationError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None



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

    use_limit = limit or settings.default_arxiv_limit
    ingested_versions: set[str] = set()
    if skip_ingested:
        sqlite_store = SQLiteStore(settings.sqlite_path)
        try:
            sqlite_store.create_schema()
            ingested_versions = sqlite_store.get_ingested_versioned_ids()
        finally:
            sqlite_store.close()

    console.print(
        "[bold]Fetching newest cs.CV papers from arXiv "
        f"(target_new={use_limit}, skip_ingested={'on' if skip_ingested else 'off'})...[/bold]"
    )
    if skip_ingested:
        console.print(f"Known ingested versions: {len(ingested_versions)}")

    fetch_stats: dict[str, int] = {}
    papers = fetch_cs_cv_papers(
        limit=use_limit,
        arxiv_api_url=settings.arxiv_api_url,
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.user_agent,
        max_retries=settings.arxiv_max_retries,
        backoff_start_seconds=settings.arxiv_backoff_start_seconds,
        backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
        skip_arxiv_id_with_version=ingested_versions if skip_ingested else None,
        stats=fetch_stats,
    )
    console.print(
        "Fetch summary: "
        f"requested={fetch_stats.get('requested', use_limit)}, "
        f"selected={fetch_stats.get('selected', len(papers))}, "
        f"skipped={fetch_stats.get('skipped', 0)}"
    )
    if not papers:
        if skip_ingested:
            console.print(
                "[yellow]No new cs.CV papers to ingest after skipping already ingested versions.[/yellow]"
            )
            return
        console.print("[yellow]No papers returned from arXiv.[/yellow]")
        raise typer.Exit(code=1)

    _run_ingest(
        papers=papers,
        metadata_json_path=settings.metadata_json_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
    )


@app.command("ingest-ids")
def ingest_ids(
    ids: list[str] = typer.Argument(
        ...,
        help="One or more arXiv IDs to ingest (example: 2104.00680 1911.11763).",
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

    console.print(f"[bold]Fetching metadata for explicit arXiv IDs ({len(ids)})...[/bold]")
    papers = fetch_papers_by_ids(
        ids=ids,
        arxiv_api_url=settings.arxiv_api_url,
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.user_agent,
        max_retries=settings.arxiv_max_retries,
        backoff_start_seconds=settings.arxiv_backoff_start_seconds,
        backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
    )
    if not papers:
        console.print("[yellow]No valid arXiv IDs provided.[/yellow]")
        raise typer.Exit(code=1)

    metadata_path = settings.metadata_dir / "arxiv_selected_ids.json"
    _run_ingest(
        papers=papers,
        metadata_json_path=metadata_path,
        force_grobid=force_grobid,
        embed_batch_size=embed_batch_size,
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
    sqlite_store = SQLiteStore(settings.sqlite_path)
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
        s2_client = SemanticScholarClient(
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
        help="Output directory for awesome_seed.jsonl.",
    ),
) -> None:
    settings = get_settings()

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
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    console.print("\n[bold green]Awesome seeding complete[/bold green]")
    console.print(f"Repos processed: {stats.repos_processed}")
    console.print(f"Total matches: {stats.total_matches}")
    console.print(f"Unique IDs: {stats.unique_ids}")
    console.print(f"JSONL output: {stats.jsonl_path}")
    console.print(f"Tier-A output: {stats.tier_a_seed_path}")

    top_repos = stats.top_repos(limit=10)
    if top_repos:
        table = Table(title="Top Repos by Match Count")
        table.add_column("Repo")
        table.add_column("Matches", justify="right")
        for repo, count in top_repos:
            table.add_row(repo, str(count))
        console.print(table)


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
    settings = get_settings()

    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.create_schema()
    qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    embed_client = OllamaEmbedClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout_seconds=settings.http_timeout_seconds,
    )

    retriever = HybridRetriever(
        embedder=embed_client,
        qdrant_store=qdrant_store,
        sqlite_store=sqlite_store,
    )

    try:
        chunks = retriever.retrieve(
            query=question,
            top_k=top_k,
            vector_k=vector_k,
            keyword_k=keyword_k,
            max_per_doc=max_per_doc,
            section_boost=section_boost,
        )
    finally:
        sqlite_store.close()

    if not chunks:
        console.print("[yellow]No chunks retrieved. Run ingest first.[/yellow]")
        raise typer.Exit(code=1)

    table = Table(title="Retrieved Chunks")
    table.add_column("Rank", justify="right")
    table.add_column("Citation")
    table.add_column("Sources")
    table.add_column("Preview")
    for idx, chunk in enumerate(chunks, start=1):
        citation = format_citation(chunk.arxiv_id, chunk.section_title)
        preview = chunk.text[:140].replace("\n", " ")
        table.add_row(str(idx), citation, ", ".join(sorted(chunk.sources)), preview)
    console.print(table)

    prompt = build_query_prompt(question, chunks)
    console.print("\n[bold]Answer prompt template:[/bold]")
    console.print(prompt)


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
        help="Answer mode: auto|single|compare|survey|implement|evidence.",
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

    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.create_schema()
    qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    embed_client = OllamaEmbedClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout_seconds=settings.http_timeout_seconds,
    )

    retriever = HybridRetriever(
        embedder=embed_client,
        qdrant_store=qdrant_store,
        sqlite_store=sqlite_store,
    )

    mode_value = mode.strip().casefold()
    if mode_value not in {"auto", "single", "compare", "survey", "implement", "evidence"}:
        console.print("[red]Invalid --mode. Use: auto|single|compare|survey|implement|evidence[/red]")
        raise typer.Exit(code=1)

    router_strategy_value = router_strategy.strip().casefold()
    if router_strategy_value not in {"rules", "llm", "hybrid"}:
        console.print("[red]Invalid --router-strategy. Use: rules|llm|hybrid[/red]")
        raise typer.Exit(code=1)
    if router_top_k <= 0:
        console.print("[red]--router-top-k must be > 0[/red]")
        raise typer.Exit(code=1)

    router_model_id = router_model or model
    entity_tokens = extract_entity_like_tokens(question)
    decision: RouteDecision
    prelim_chunks: list[RetrievedChunk]
    maybe_relevant: list[RetrievedChunk]
    final_k = k if k is not None else 8
    final_max_per_doc = max_per_doc if max_per_doc is not None else 4
    try:
        prelim_chunks, maybe_relevant = retrieve_for_answer(
            retriever=retriever,
            question=question,
            k=router_top_k,
            max_per_doc=2,
            section_boost=section_boost,
            entity_tokens=entity_tokens,
        )

        if entity_tokens and 0 < len(prelim_chunks) < router_top_k:
            console.print(f"[yellow]Only {len(prelim_chunks)} relevant sources found.[/yellow]")

        if retriever._is_irrelevant_result(
            question,
            prelim_chunks[: max(3, len(prelim_chunks))],
            settings.relevance_vector_threshold,
        ):
            console.print("Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763")
            if maybe_relevant:
                table = Table(title="Maybe Relevant (Top 3)")
                table.add_column("Rank", justify="right")
                table.add_column("Citation")
                table.add_column("Preview")
                for idx, chunk in enumerate(maybe_relevant, start=1):
                    citation = format_citation(chunk.arxiv_id, chunk.section_title)
                    preview = chunk.text[:160].replace("\n", " ")
                    table.add_row(str(idx), citation, preview)
                console.print(table)
            raise typer.Exit(code=1)

        if not prelim_chunks:
            console.print("[red]No sources retrieved for this question. Run ingest first or broaden the query.[/red]")
            raise typer.Exit(code=1)

        if mode_value == "auto":
            decision = route_answer_mode(
                question=question,
                prelim_chunks=prelim_chunks,
                model_id=router_model_id,
                strategy=router_strategy_value,
            )
        else:
            forced_mode = mode_from_value(mode_value)
            decision = make_route_decision(
                forced_mode,
                notes=f"Manual mode override: {forced_mode.value}.",
                confidence=1.0,
            )

        final_k = k if k is not None else decision.k
        final_max_per_doc = max_per_doc if max_per_doc is not None else decision.max_per_doc
        effective_section_boost = max(section_boost, decision.section_boost_hint)

        chunks, _ = retrieve_for_answer(
            retriever=retriever,
            question=question,
            k=final_k,
            max_per_doc=final_max_per_doc,
            section_boost=effective_section_boost,
            entity_tokens=entity_tokens,
        )
    finally:
        sqlite_store.close()

    if entity_tokens and 0 < len(chunks) < final_k:
        console.print(f"[yellow]Only {len(chunks)} relevant sources found.[/yellow]")

    if not chunks:
        console.print("[red]No sources retrieved for this question. Run ingest first or broaden the query.[/red]")
        raise typer.Exit(code=1)

    if decision.mode is AnswerMode.COMPARE:
        cross_doc_decision = enforce_cross_doc_support(question, chunks)
        chunks = cross_doc_decision.filtered_chunks
        for warning in cross_doc_decision.warnings:
            console.print(f"[yellow]{warning}[/yellow]")
        if cross_doc_decision.should_refuse:
            console.print("[red]Refusing to answer comparison without sufficient cross-paper coverage.[/red]")
            raise typer.Exit(code=1)

    if decision.preface:
        console.print(f"[yellow]{decision.preface}[/yellow]")

    if show_sources:
        table = Table(title="Retrieved Sources")
        table.add_column("Source")
        table.add_column("arXiv ID")
        table.add_column("Title")
        table.add_column("Section")
        table.add_column("Preview")
        for idx, chunk in enumerate(chunks, start=1):
            preview = chunk.text[:140].replace("\n", " ")
            section = chunk.section_title.strip() or "Untitled"
            table.add_row(f"S{idx}", chunk.arxiv_id, chunk.title, section, preview)
        console.print(table)

    prompt = build_mode_answer_prompt(
        decision.mode,
        question,
        chunks,
        route_preface=decision.preface,
    )
    draft_text = _mlx_generate_or_exit(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    valid, reason = validate_answer_citations(draft_text, len(chunks))
    answer_text = draft_text
    if not valid:
        console.print("[yellow]Draft failed citation check; attempting repair…[/yellow]")
        repair_prompt = build_mode_repair_prompt(
            decision.mode,
            question,
            chunks,
            draft_text,
            route_preface=decision.preface,
        )
        answer_text = _mlx_generate_or_exit(
            model=model,
            prompt=repair_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        valid, reason = validate_answer_citations(answer_text, len(chunks))
        if not valid:
            console.print(f"[red]Refusing answer: citation grounding check failed ({reason})[/red]")
            console.print("\n[bold]Draft[/bold]")
            console.print(draft_text)
            if no_refuse:
                console.print(
                    "\n[bold red]WARNING: --no-refuse enabled."
                    " Returning draft despite failed citation validation.[/bold red]"
                )
                console.print("\n[bold]Answer (Unvalidated Draft)[/bold]")
                console.print(draft_text)
                return
            raise typer.Exit(code=1)

    console.print("\n[bold]Answer[/bold]")
    console.print(answer_text)


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
    if not suite.exists():
        console.print(f"[red]Eval suite not found: {suite}[/red]")
        raise typer.Exit(code=1)

    try:
        cases = _load_eval_cases(suite)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if not cases:
        console.print(f"[yellow]Eval suite has no cases: {suite}[/yellow]")
        raise typer.Exit(code=1)

    settings = get_settings()
    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.create_schema()
    qdrant_store = QdrantStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    embed_client = OllamaEmbedClient(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout_seconds=settings.http_timeout_seconds,
    )
    retriever = HybridRetriever(
        embedder=embed_client,
        qdrant_store=qdrant_store,
        sqlite_store=sqlite_store,
    )

    results: list[EvalCaseResult] = []
    try:
        for idx, case in enumerate(cases, start=1):
            entity_tokens = extract_entity_like_tokens(case.question)
            chunks, _ = retrieve_for_answer(
                retriever=retriever,
                question=case.question,
                k=k,
                max_per_doc=max_per_doc,
                section_boost=section_boost,
                entity_tokens=entity_tokens,
            )

            reasons: list[str] = []
            if len(chunks) < case.min_sources:
                reasons.append(f"sources={len(chunks)} < min_sources={case.min_sources}")
            threshold = settings.relevance_vector_threshold
            if retriever._is_irrelevant_result(case.question, chunks[: max(3, len(chunks))], threshold):
                reasons.append("retrieval deemed irrelevant")
            if not _has_eval_constraint_match(chunks, case):
                reasons.append("no chunk matched must_include constraints")

            if reasons:
                results.append(
                    EvalCaseResult(
                        index=idx,
                        question=case.question,
                        status=False,
                        retrieved_sources=len(chunks),
                        note="; ".join(reasons),
                    )
                )
                continue

            prompt = build_strict_answer_prompt(case.question, chunks)
            answer_text = _mlx_generate_or_exit(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                top_p=1.0,
                seed=0,
            )
            citations_ok, citation_reason = validate_answer_citations(answer_text, len(chunks))
            results.append(
                EvalCaseResult(
                    index=idx,
                    question=case.question,
                    status=citations_ok,
                    retrieved_sources=len(chunks),
                    note="" if citations_ok else citation_reason,
                )
            )
    finally:
        sqlite_store.close()

    table = Table(title=f"Eval Results ({sum(1 for r in results if r.status)}/{len(results)} passed)")
    table.add_column("#", justify="right")
    table.add_column("Status")
    table.add_column("Sources", justify="right")
    table.add_column("Question")
    table.add_column("Note")
    for result in results:
        table.add_row(
            str(result.index),
            "PASS" if result.status else "FAIL",
            str(result.retrieved_sources),
            result.question,
            result.note,
        )
    console.print(table)

    if any(not result.status for result in results):
        raise typer.Exit(code=1)


@app.command()
def stats(
    top_venues: int = typer.Option(
        10,
        "--top-venues",
        help="Show top-N venues from paper_metrics (0 to disable).",
    )
) -> None:
    settings = get_settings()
    settings.ensure_directories()

    if not settings.sqlite_path.exists():
        console.print(f"[red]SQLite database not found: {settings.sqlite_path}[/red]")
        raise typer.Exit(code=1)

    sqlite_store = SQLiteStore(settings.sqlite_path)
    sqlite_store.create_schema()

    def scalar(query: str, params: tuple[object, ...] = ()) -> int:
        row = sqlite_store.conn.execute(query, params).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    try:
        papers_count = scalar("SELECT COUNT(*) FROM papers")
        chunks_count = scalar("SELECT COUNT(*) FROM chunks")
        chunk_docs_count = scalar("SELECT COUNT(DISTINCT arxiv_id) FROM chunks")
        metrics_count = scalar("SELECT COUNT(*) FROM paper_metrics")
        papers_without_metrics = scalar(
            """
            SELECT COUNT(*)
            FROM papers p
            LEFT JOIN paper_metrics m ON p.arxiv_id = m.arxiv_id
            WHERE m.arxiv_id IS NULL
            """
        )
        indexed_pdf_rows = sqlite_store.conn.execute(
            """
            SELECT pdf_path
            FROM papers
            WHERE pdf_path IS NOT NULL
              AND TRIM(pdf_path) != ''
            """
        ).fetchall()
        indexed_pdf_names = {Path(str(row[0])).name for row in indexed_pdf_rows if row[0]}

        tier_rows = sqlite_store.conn.execute(
            """
            SELECT tier, COUNT(*) AS count
            FROM paper_metrics
            GROUP BY tier
            ORDER BY tier
            """
        ).fetchall()
        venue_rows = []
        if top_venues > 0:
            venue_rows = sqlite_store.conn.execute(
                """
                SELECT venue, COUNT(*) AS count
                FROM paper_metrics
                WHERE venue IS NOT NULL
                  AND TRIM(venue) != ''
                GROUP BY venue
                ORDER BY count DESC
                LIMIT ?
                """,
                (top_venues,),
            ).fetchall()
    finally:
        sqlite_store.close()

    pdf_files = list(settings.pdf_dir.glob("*.pdf"))
    tei_files = list(settings.tei_dir.glob("*.tei.xml"))
    pdf_file_names = {path.name for path in pdf_files}
    orphan_pdf_files = pdf_file_names - indexed_pdf_names

    table = Table(title="Database Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("sqlite path", str(settings.sqlite_path))
    table.add_row("papers rows", str(papers_count))
    table.add_row("chunks rows", str(chunks_count))
    table.add_row("chunk docs (distinct arxiv_id)", str(chunk_docs_count))
    table.add_row("paper_metrics rows", str(metrics_count))
    table.add_row("papers without metrics", str(papers_without_metrics))
    table.add_row("pdf files on disk", str(len(pdf_files)))
    table.add_row("tei files on disk", str(len(tei_files)))
    table.add_row("indexed pdf paths in papers", str(len(indexed_pdf_names)))
    table.add_row("orphan pdf files (disk - indexed)", str(len(orphan_pdf_files)))
    console.print(table)

    if tier_rows:
        tier_table = Table(title="Tier Distribution")
        tier_table.add_column("Tier", justify="right")
        tier_table.add_column("Count", justify="right")
        for row in tier_rows:
            tier_table.add_row(str(int(row["tier"])), str(int(row["count"])))
        console.print(tier_table)

    if venue_rows:
        venue_table = Table(title=f"Top Venues (top {top_venues})")
        venue_table.add_column("Venue")
        venue_table.add_column("Count", justify="right")
        for row in venue_rows:
            venue_table.add_row(str(row["venue"]), str(int(row["count"])))
        console.print(venue_table)


@app.command()
def doctor(
    qdrant_test_point: bool = typer.Option(
        False,
        help="Insert one test point into Qdrant.",
    )
) -> None:
    settings = get_settings()
    table = Table(title="Service Health")
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Version/Detail")

    # Qdrant checks
    try:
        root = httpx.get(settings.qdrant_url, timeout=5.0)
        root.raise_for_status()
        version = root.json().get("version", "unknown") if "json" in dir(root) else "unknown"

        qdrant_store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
        qdrant_store.client.get_collections()
        table.add_row("Qdrant", "ok", f"version={version}")

        if qdrant_test_point:
            doctor_collection = f"{settings.qdrant_collection}_doctor"
            doctor_store = QdrantStore(settings.qdrant_url, doctor_collection)
            doctor_store.ensure_collection(3)
            point_id = str(uuid.uuid4())
            doctor_store.upsert(
                [
                    VectorPoint(
                        point_id=point_id,
                        vector=[0.1, 0.2, 0.3],
                        payload={"kind": "doctor", "chunk_id": point_id},
                    )
                ]
            )
            table.add_row("Qdrant test insert", "ok", f"collection={doctor_collection}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Qdrant", "fail", str(exc))

    # GROBID checks
    try:
        alive = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/isalive", timeout=5.0)
        alive.raise_for_status()
        isalive_text = alive.text.strip()

        version_resp = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/version", timeout=5.0)
        grobid_version = version_resp.text.strip() if version_resp.status_code < 500 else "unknown"
        table.add_row("GROBID", "ok", f"alive={isalive_text}; version={grobid_version}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("GROBID", "fail", str(exc))

    # Ollama checks
    try:
        version_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/version", timeout=5.0)
        version_resp.raise_for_status()
        version = version_resp.json().get("version", "unknown")

        tags_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        tags_resp.raise_for_status()
        model_count = len(tags_resp.json().get("models", []))
        table.add_row("Ollama", "ok", f"version={version}; models={model_count}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Ollama", "fail", str(exc))

    console.print(table)


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host."),
    port: int = typer.Option(8000, help="Bind port."),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
) -> None:
    """Start the web UI server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]Web dependencies not installed. Run: uv sync --extra web[/red]")
        raise typer.Exit(code=1) from None

    uvicorn.run("cv_rag.api.app:create_app", host=host, port=port, reload=reload, factory=True)


if __name__ == "__main__":
    app()
