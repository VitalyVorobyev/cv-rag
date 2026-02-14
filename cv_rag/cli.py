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
    build_repair_prompt,
    build_strict_answer_prompt,
    is_comparison_question,
    retrieve_for_answer,
    top_doc_source_counts,
    validate_answer_citations,
)
from cv_rag.arxiv_sync import (
    PaperMetadata,
    fetch_cs_cv_papers,
    fetch_papers_by_ids,
)
from cv_rag.config import get_settings
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.exceptions import GenerationError
from cv_rag.ingest import IngestPipeline
from cv_rag.llm import mlx_generate
from cv_rag.qdrant_store import QdrantStore, VectorPoint
from cv_rag.retrieve import (
    HybridRetriever,
    RetrievedChunk,
    extract_entity_like_tokens,
    format_citation,
)
from cv_rag.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)

app = typer.Typer(help="Local CV papers RAG MVP")
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
    k: int = typer.Option(12, "--k", help="Number of sources to include after retrieval merge."),
    model: str = typer.Option(..., "--model", help="MLX model ID, local path, or HF repo."),
    max_tokens: int = typer.Option(600, "--max-tokens", help="Maximum tokens to generate."),
    temperature: float = typer.Option(0.2, "--temperature", help="Sampling temperature."),
    top_p: float = typer.Option(0.9, "--top-p", help="Sampling top-p."),
    seed: int | None = typer.Option(None, "--seed", help="Optional random seed."),
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

    entity_tokens = extract_entity_like_tokens(question)
    try:
        chunks, maybe_relevant = retrieve_for_answer(
            retriever=retriever,
            question=question,
            k=k,
            max_per_doc=max_per_doc,
            section_boost=section_boost,
            entity_tokens=entity_tokens,
        )
    finally:
        sqlite_store.close()

    if entity_tokens and 0 < len(chunks) < k:
        console.print(f"[yellow]Only {len(chunks)} relevant sources found.[/yellow]")

    if retriever._is_irrelevant_result(question, chunks[: max(3, len(chunks))], settings.relevance_vector_threshold):
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

    if not chunks:
        console.print("[red]No sources retrieved for this question. Run ingest first or broaden the query.[/red]")
        raise typer.Exit(code=1)

    top_doc_counts = top_doc_source_counts(chunks)
    enough_cross_doc_support = len(top_doc_counts) >= 2 and all(count >= 2 for _, count in top_doc_counts)
    if not enough_cross_doc_support:
        console.print(
            "[yellow]Warning: need at least 2 sources from each of the top 2 papers"
            " (by score) for robust comparison grounding.[/yellow]"
        )
        if top_doc_counts:
            details = ", ".join(f"{arxiv_id} ({count})" for arxiv_id, count in top_doc_counts)
            console.print(f"[yellow]Current top-paper source counts: {details}[/yellow]")
        if is_comparison_question(question):
            console.print("[red]Refusing to answer comparison without sufficient cross-paper coverage.[/red]")
            raise typer.Exit(code=1)

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

    prompt = build_strict_answer_prompt(question, chunks)
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
        repair_prompt = build_repair_prompt(question, chunks, draft_text)
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


if __name__ == "__main__":
    app()
