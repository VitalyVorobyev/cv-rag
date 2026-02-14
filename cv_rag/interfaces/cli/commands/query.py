from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from cv_rag.answer.citations import build_query_prompt
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.retrieval.hybrid import HybridRetriever, format_citation
from cv_rag.shared.settings import Settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore


def run_query_command(
    *,
    settings: Settings,
    console: Console,
    question: str,
    top_k: int,
    vector_k: int,
    keyword_k: int,
    max_per_doc: int,
    section_boost: float,
    qdrant_store_cls: type[QdrantStore] = QdrantStore,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
    embed_client_cls: type[OllamaEmbedClient] = OllamaEmbedClient,
    retriever_cls: type[HybridRetriever] = HybridRetriever,
) -> None:
    sqlite_store = sqlite_store_cls(settings.sqlite_path)
    sqlite_store.create_schema()
    qdrant_store = qdrant_store_cls(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection,
    )
    embed_client = embed_client_cls(
        base_url=settings.ollama_url,
        model=settings.ollama_model,
        timeout_seconds=settings.http_timeout_seconds,
    )

    retriever = retriever_cls(
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
