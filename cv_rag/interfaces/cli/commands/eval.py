from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import typer
import yaml
from pydantic import BaseModel, Field, ValidationError
from rich.console import Console
from rich.table import Table

from cv_rag.answer.citations import (
    build_strict_answer_prompt,
    retrieve_for_answer,
    validate_answer_citations,
)
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.retrieval.hybrid import HybridRetriever, RetrievedChunk
from cv_rag.retrieval.relevance import extract_entity_like_tokens
from cv_rag.shared.errors import GenerationError
from cv_rag.shared.settings import Settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore


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


def load_eval_cases(suite_path: Path) -> list[EvalCase]:
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


def has_eval_constraint_match(chunks: list[RetrievedChunk], case: EvalCase) -> bool:
    if not case.must_include_arxiv_ids and not case.must_include_tokens:
        return True
    return any(_chunk_matches_eval_constraints(chunk, case) for chunk in chunks)


def mlx_generate_or_exit(
    *,
    console: Console,
    mlx_generate_fn: object,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> str:
    try:
        return mlx_generate_fn(  # type: ignore[operator]
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


def run_eval_command(
    *,
    settings: Settings,
    console: Console,
    suite: Path,
    model: str,
    k: int,
    max_per_doc: int,
    section_boost: float,
    max_tokens: int,
    mlx_generate_fn: object,
    qdrant_store_cls: type[QdrantStore] = QdrantStore,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
    embed_client_cls: type[OllamaEmbedClient] = OllamaEmbedClient,
    retriever_cls: type[HybridRetriever] = HybridRetriever,
) -> None:
    if not suite.exists():
        console.print(f"[red]Eval suite not found: {suite}[/red]")
        raise typer.Exit(code=1)

    try:
        cases = load_eval_cases(suite)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from None

    if not cases:
        console.print(f"[yellow]Eval suite has no cases: {suite}[/yellow]")
        raise typer.Exit(code=1)

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
            if not has_eval_constraint_match(chunks, case):
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
            answer_text = mlx_generate_or_exit(
                console=console,
                mlx_generate_fn=mlx_generate_fn,
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
