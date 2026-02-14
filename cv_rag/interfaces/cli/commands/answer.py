from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from cv_rag.answer.models import AnswerRunRequest
from cv_rag.answer.service import AnswerService
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.retrieval.hybrid import HybridRetriever
from cv_rag.shared.errors import CitationValidationError
from cv_rag.shared.settings import Settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore


def run_answer_command(
    *,
    settings: Settings,
    console: Console,
    question: str,
    model: str,
    mode: str,
    router_model: str | None,
    router_strategy: str,
    router_top_k: int,
    k: int | None,
    max_per_doc: int | None,
    section_boost: float,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    show_sources: bool,
    no_refuse: bool,
    mlx_generate_fn: object,
    qdrant_store_cls: type[QdrantStore] = QdrantStore,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
    embed_client_cls: type[OllamaEmbedClient] = OllamaEmbedClient,
    retriever_cls: type[HybridRetriever] = HybridRetriever,
    answer_service_cls: type[AnswerService] = AnswerService,
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

    request = AnswerRunRequest(
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
        no_refuse=no_refuse,
    )
    service = answer_service_cls(
        retriever=retriever,
        settings=settings,
        generate_fn=mlx_generate_fn,  # type: ignore[arg-type]
    )

    try:
        result = service.run(request)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except LookupError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None
    except CitationValidationError as exc:
        console.print(f"[red]Refusing answer: citation grounding check failed ({exc.reason})[/red]")
        console.print("\n[bold]Draft[/bold]")
        console.print(exc.draft)
        if no_refuse:
            console.print(
                "\n[bold red]WARNING: --no-refuse enabled."
                " Returning draft despite failed citation validation.[/bold red]"
            )
            console.print("\n[bold]Answer (Unvalidated Draft)[/bold]")
            console.print(exc.draft)
            return
        raise typer.Exit(code=1) from None
    finally:
        sqlite_store.close()

    for warning in result.warnings:
        console.print(f"[yellow]{warning}[/yellow]")

    if result.route.preface:
        console.print(f"[yellow]{result.route.preface}[/yellow]")

    if show_sources:
        table = Table(title="Retrieved Sources")
        table.add_column("Source")
        table.add_column("arXiv ID")
        table.add_column("Title")
        table.add_column("Section")
        table.add_column("Preview")
        for idx, chunk in enumerate(result.sources, start=1):
            preview = chunk.text[:140].replace("\n", " ")
            section = chunk.section_title.strip() or "Untitled"
            table.add_row(f"S{idx}", chunk.arxiv_id, chunk.title, section, preview)
        console.print(table)

    console.print("\n[bold]Answer[/bold]")
    console.print(result.answer)
