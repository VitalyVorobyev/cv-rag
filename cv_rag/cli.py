from __future__ import annotations

from pathlib import Path
import uuid

import httpx
from rich.console import Console
from rich.table import Table
import typer

from cv_rag.arxiv_sync import download_pdf, fetch_cs_cv_papers, write_metadata_json
from cv_rag.chunking import chunk_sections
from cv_rag.config import get_settings
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.grobid_client import pdf_to_tei
from cv_rag.qdrant_store import QdrantStore, VectorPoint
from cv_rag.retrieve import HybridRetriever, build_answer_prompt, format_citation
from cv_rag.sqlite_store import SQLiteStore
from cv_rag.tei_extract import extract_sections


app = typer.Typer(help="Local CV papers RAG MVP")
console = Console()


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
    use_batch_size = embed_batch_size or settings.embed_batch_size

    console.print(
        f"[bold]Fetching newest cs.CV papers from arXiv (limit={use_limit})...[/bold]"
    )
    papers = fetch_cs_cv_papers(
        limit=use_limit,
        arxiv_api_url=settings.arxiv_api_url,
        timeout_seconds=settings.http_timeout_seconds,
        user_agent=settings.user_agent,
    )
    if not papers:
        console.print("[yellow]No papers returned from arXiv.[/yellow]")
        raise typer.Exit(code=1)

    write_metadata_json(papers, settings.metadata_json_path)
    console.print(f"Saved metadata: {settings.metadata_json_path}")

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
def query(
    question: str = typer.Argument(..., help="Question to ask against local index."),
    top_k: int = typer.Option(8, help="Final number of merged chunks to keep."),
    vector_k: int = typer.Option(12, help="Top-k vector hits from Qdrant."),
    keyword_k: int = typer.Option(12, help="Top-k keyword hits from SQLite FTS."),
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
