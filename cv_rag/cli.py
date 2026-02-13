from __future__ import annotations

from pathlib import Path
import re
import subprocess
import uuid

import httpx
from rich.console import Console
from rich.table import Table
import typer

from cv_rag.arxiv_sync import (
    PaperMetadata,
    download_pdf,
    fetch_cs_cv_papers,
    fetch_papers_by_ids,
    write_metadata_json,
)
from cv_rag.chunking import chunk_sections
from cv_rag.config import get_settings
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.grobid_client import pdf_to_tei
from cv_rag.qdrant_store import QdrantStore, VectorPoint
from cv_rag.retrieve import (
    HybridRetriever,
    NoRelevantSourcesError,
    RetrievedChunk,
    build_answer_prompt,
    build_strict_answer_prompt,
    format_citation,
)
from cv_rag.sqlite_store import SQLiteStore
from cv_rag.tei_extract import extract_sections


app = typer.Typer(help="Local CV papers RAG MVP")
console = Console()
COMPARISON_QUERY_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|better than|worse than)\b",
    re.IGNORECASE,
)
CITATION_REF_RE = re.compile(r"\[S\d+\]")


def _load_tei_or_parse(pdf_path: Path, tei_path: Path, force: bool) -> str:
    settings = get_settings()
    if tei_path.exists() and not force:
        return tei_path.read_text(encoding="utf-8")

    tei_xml = pdf_to_tei(
        pdf_path=pdf_path,
        grobid_url=settings.grobid_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.grobid_max_retries,
        backoff_start_seconds=settings.grobid_backoff_start_seconds,
        backoff_cap_seconds=settings.grobid_backoff_cap_seconds,
    )
    tei_path.write_text(tei_xml, encoding="utf-8")
    return tei_xml


def _is_comparison_question(question: str) -> bool:
    return COMPARISON_QUERY_RE.search(question) is not None


def _top_doc_source_counts(chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    if not chunks:
        return []
    best_scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.arxiv_id] = counts.get(chunk.arxiv_id, 0) + 1
        best_scores[chunk.arxiv_id] = max(best_scores.get(chunk.arxiv_id, float("-inf")), chunk.fused_score)
    top_docs = sorted(best_scores.keys(), key=lambda arxiv_id: best_scores[arxiv_id], reverse=True)[:2]
    return [(arxiv_id, counts[arxiv_id]) for arxiv_id in top_docs]


def _run_mlx_generate(
    *,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> str:
    command = [
        "mlx_lm.generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--temp",
        str(temperature),
        "--top-p",
        str(top_p),
        "--verbose",
        "False",
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        console.print("[red]`mlx_lm.generate` was not found in PATH.[/red]")
        raise typer.Exit(code=1) from None

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if detail:
            console.print(f"[red]MLX generation failed: {detail}[/red]")
        else:
            console.print(f"[red]MLX generation failed with exit code {result.returncode}.[/red]")
        raise typer.Exit(code=1)

    answer_text = result.stdout.strip()
    if not answer_text:
        console.print("[red]MLX generation returned an empty answer.[/red]")
        raise typer.Exit(code=1)
    return answer_text


def _ingest_papers(
    papers: list[PaperMetadata],
    metadata_json_path: Path,
    force_grobid: bool,
    embed_batch_size: int | None,
) -> None:
    settings = get_settings()
    use_batch_size = embed_batch_size or settings.embed_batch_size

    if not papers:
        console.print("[yellow]No papers to ingest.[/yellow]")
        raise typer.Exit(code=1)

    write_metadata_json(papers, metadata_json_path)
    console.print(f"Saved metadata: {metadata_json_path}")

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

    total_chunks = 0
    collection_initialized = False

    try:
        for idx, paper in enumerate(papers, start=1):
            console.print(f"[{idx}/{len(papers)}] {paper.arxiv_id_with_version} - {paper.title}")
            try:
                pdf_path = download_pdf(
                    paper=paper,
                    pdf_dir=settings.pdf_dir,
                    timeout_seconds=settings.http_timeout_seconds,
                    user_agent=settings.user_agent,
                )
                tei_path = settings.tei_dir / f"{paper.safe_file_stem()}.tei.xml"
                tei_xml = _load_tei_or_parse(pdf_path=pdf_path, tei_path=tei_path, force=force_grobid)

                sections = extract_sections(tei_xml)
                chunks = chunk_sections(
                    sections,
                    max_chars=settings.chunk_max_chars,
                    overlap_chars=settings.chunk_overlap_chars,
                )
                if not chunks:
                    console.print(
                        f"[yellow]  Skipping {paper.arxiv_id_with_version}: no text chunks extracted.[/yellow]"
                    )
                    continue

                chunk_texts = [c.text for c in chunks]
                vectors = embed_client.embed_in_batches(chunk_texts, batch_size=use_batch_size)
                if len(vectors) != len(chunks):
                    raise RuntimeError(
                        f"Embedding count mismatch for {paper.arxiv_id}: "
                        f"{len(vectors)} vectors for {len(chunks)} chunks"
                    )

                if vectors and not collection_initialized:
                    qdrant_store.ensure_collection(len(vectors[0]))
                    collection_initialized = True

                sqlite_store.upsert_paper(paper=paper, pdf_path=pdf_path, tei_path=tei_path)

                sqlite_rows: list[dict[str, object]] = []
                points: list[VectorPoint] = []
                for chunk, vector in zip(chunks, vectors):
                    chunk_id = f"{paper.arxiv_id}:{chunk.chunk_index}"
                    sqlite_row = {
                        "chunk_id": chunk_id,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "section_title": chunk.section_title,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                    }
                    sqlite_rows.append(sqlite_row)

                    payload = {
                        "chunk_id": chunk_id,
                        "arxiv_id": paper.arxiv_id,
                        "title": paper.title,
                        "section_title": chunk.section_title,
                        "text": chunk.text,
                        "citation": format_citation(paper.arxiv_id, chunk.section_title),
                    }
                    points.append(VectorPoint(point_id=chunk_id, vector=vector, payload=payload))

                sqlite_store.upsert_chunks(sqlite_rows)
                qdrant_store.upsert(points)
                total_chunks += len(chunks)
                console.print(f"  Ingested {len(chunks)} chunks")
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]  Failed: {exc}[/red]")
    finally:
        sqlite_store.close()

    console.print("\n[bold green]Ingest complete[/bold green]")
    console.print(f"Papers processed: {len(papers)}")
    console.print(f"Total chunks upserted: {total_chunks}")
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

    console.print(
        f"[bold]Fetching newest cs.CV papers from arXiv (limit={use_limit})...[/bold]"
    )
    papers = fetch_cs_cv_papers(
        limit=use_limit,
        arxiv_api_url=settings.arxiv_api_url,
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.user_agent,
        max_retries=settings.arxiv_max_retries,
        backoff_start_seconds=settings.arxiv_backoff_start_seconds,
        backoff_cap_seconds=settings.arxiv_backoff_cap_seconds,
    )
    if not papers:
        console.print("[yellow]No papers returned from arXiv.[/yellow]")
        raise typer.Exit(code=1)

    _ingest_papers(
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
    _ingest_papers(
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
            max_chunks_per_doc=max_per_doc,
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

    prompt = build_answer_prompt(question, chunks)
    console.print("\n[bold]Answer prompt template:[/bold]")
    console.print(prompt)


@app.command()
def answer(
    question: str = typer.Argument(..., help="Question to answer from local index."),
    k: int = typer.Option(10, "--k", help="Number of sources to include."),
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
        0.0,
        "--section-boost",
        help="Additive score boost for method/training-oriented section/title matches.",
    ),
    show_sources: bool = typer.Option(
        False,
        "--show-sources",
        help="Print retrieved sources before generating the final answer.",
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
            top_k=k,
            vector_k=max(k, 12),
            keyword_k=max(k, 12),
            require_relevance=True,
            max_chunks_per_doc=max_per_doc,
            section_boost=section_boost,
        )
    except NoRelevantSourcesError as exc:
        console.print("Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763")
        if exc.candidates:
            table = Table(title="Maybe Relevant (Top 3)")
            table.add_column("Rank", justify="right")
            table.add_column("Citation")
            table.add_column("Preview")
            for idx, chunk in enumerate(exc.candidates[:3], start=1):
                citation = format_citation(chunk.arxiv_id, chunk.section_title)
                preview = chunk.text[:160].replace("\n", " ")
                table.add_row(str(idx), citation, preview)
            console.print(table)
        raise typer.Exit(code=1)
    finally:
        sqlite_store.close()

    if not chunks:
        console.print("[red]No sources retrieved for this question. Run ingest first or broaden the query.[/red]")
        raise typer.Exit(code=1)

    top_doc_counts = _top_doc_source_counts(chunks)
    enough_cross_doc_support = len(top_doc_counts) >= 2 and all(count >= 2 for _, count in top_doc_counts)
    if not enough_cross_doc_support:
        console.print(
            "[yellow]Warning: need at least 2 sources from each of the top 2 papers (by score) for robust comparison grounding.[/yellow]"
        )
        if top_doc_counts:
            details = ", ".join(f"{arxiv_id} ({count})" for arxiv_id, count in top_doc_counts)
            console.print(f"[yellow]Current top-paper source counts: {details}[/yellow]")
        if _is_comparison_question(question):
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
    answer_text = _run_mlx_generate(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    min_citation_count = min(2, len(chunks))
    if len(CITATION_REF_RE.findall(answer_text)) < min_citation_count:
        revised_prompt = (
            f"{prompt}\n\nYou forgot citations; revise.\n"
            "Return a revised answer with inline citations like [S1][S3] for non-trivial claims."
        )
        answer_text = _run_mlx_generate(
            model=model,
            prompt=revised_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

    console.print("\n[bold]Answer[/bold]")
    console.print(answer_text)


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
