from __future__ import annotations

import logging
import time
from collections.abc import Generator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from cv_rag.answer import (
    enforce_cross_doc_support,
    retrieve_for_answer,
    validate_answer_citations,
)
from cv_rag.api.deps import get_app_settings, get_retriever
from cv_rag.api.schemas import AnswerRequest, AnswerResponse, ChunkResponse, RouteInfo
from cv_rag.api.streaming import mlx_generate_stream, sse_event
from cv_rag.config import Settings
from cv_rag.exceptions import GenerationError
from cv_rag.llm import mlx_generate
from cv_rag.prompts_answer import build_prompt as build_mode_answer_prompt
from cv_rag.prompts_answer import build_repair_prompt as build_mode_repair_prompt
from cv_rag.retrieve import HybridRetriever, RetrievedChunk, extract_entity_like_tokens
from cv_rag.routing import AnswerMode, make_route_decision, mode_from_value
from cv_rag.routing import route as route_answer_mode

logger = logging.getLogger(__name__)

router = APIRouter(tags=["answer"])


def _chunks_to_response(chunks: list[RetrievedChunk]) -> list[ChunkResponse]:
    return [
        ChunkResponse(
            chunk_id=c.chunk_id,
            arxiv_id=c.arxiv_id,
            title=c.title,
            section_title=c.section_title,
            text=c.text,
            fused_score=c.fused_score,
            vector_score=c.vector_score,
            keyword_score=c.keyword_score,
            sources=sorted(c.sources),
        )
        for c in chunks
    ]


def _route_to_info(decision: object) -> RouteInfo:
    return RouteInfo(
        mode=str(decision.mode.value),  # type: ignore[union-attr]
        targets=list(decision.targets),  # type: ignore[union-attr]
        k=decision.k,  # type: ignore[union-attr]
        max_per_doc=decision.max_per_doc,  # type: ignore[union-attr]
        confidence=decision.confidence,  # type: ignore[union-attr]
        notes=decision.notes,  # type: ignore[union-attr]
        preface=decision.preface,  # type: ignore[union-attr]
    )


@router.post("/answer")
def answer_question(
    body: AnswerRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_app_settings),
) -> StreamingResponse:
    def generate() -> Generator[str, None, None]:
        t0 = time.perf_counter()

        try:
            # Step 1: Entity extraction + preliminary retrieval
            entity_tokens = extract_entity_like_tokens(body.question)
            prelim_chunks, maybe_relevant = retrieve_for_answer(
                retriever=retriever,
                question=body.question,
                k=12,
                max_per_doc=2,
                section_boost=body.section_boost,
                entity_tokens=entity_tokens,
            )

            # Step 2: Relevance check
            if retriever._is_irrelevant_result(
                body.question,
                prelim_chunks[: max(3, len(prelim_chunks))],
                settings.relevance_vector_threshold,
            ):
                yield sse_event("error", {
                    "message": "Not found in indexed corpus.",
                    "maybe_relevant": [
                        {"arxiv_id": c.arxiv_id, "section_title": c.section_title, "preview": c.text[:160]}
                        for c in maybe_relevant
                    ],
                })
                return

            if not prelim_chunks:
                yield sse_event("error", {"message": "No sources retrieved. Run ingest first."})
                return

            # Step 3: Routing
            mode_value = body.mode.strip().casefold()
            if mode_value == "auto":
                decision = route_answer_mode(
                    question=body.question,
                    prelim_chunks=prelim_chunks,
                    model_id=body.model,
                    strategy=body.router_strategy,
                )
            else:
                forced_mode = mode_from_value(mode_value)
                decision = make_route_decision(
                    forced_mode,
                    notes=f"Manual mode override: {forced_mode.value}.",
                    confidence=1.0,
                )

            route_info = _route_to_info(decision)
            yield sse_event("route", route_info.model_dump())

            # Step 4: Final retrieval with route-determined parameters
            final_k = body.k if body.k is not None else decision.k
            final_max_per_doc = body.max_per_doc if body.max_per_doc is not None else decision.max_per_doc
            effective_section_boost = max(body.section_boost, decision.section_boost_hint)

            chunks, _ = retrieve_for_answer(
                retriever=retriever,
                question=body.question,
                k=final_k,
                max_per_doc=final_max_per_doc,
                section_boost=effective_section_boost,
                entity_tokens=entity_tokens,
            )

            if not chunks:
                yield sse_event("error", {"message": "No sources retrieved after routing."})
                return

            # Step 5: Cross-doc enforcement for compare mode
            if decision.mode is AnswerMode.COMPARE:
                cross_doc = enforce_cross_doc_support(body.question, chunks)
                chunks = cross_doc.filtered_chunks
                if cross_doc.should_refuse:
                    yield sse_event("error", {
                        "message": "Insufficient cross-paper coverage for comparison.",
                        "warnings": cross_doc.warnings,
                    })
                    return

            # Emit sources
            yield sse_event("sources", [c.model_dump() for c in _chunks_to_response(chunks)])

            # Step 6: Build prompt and generate
            prompt = build_mode_answer_prompt(
                decision.mode,
                body.question,
                chunks,
                route_preface=decision.preface,
            )

            # Stream generation
            collected_tokens: list[str] = []
            try:
                for token_chunk in mlx_generate_stream(
                    model=body.model,
                    prompt=prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    seed=body.seed,
                ):
                    collected_tokens.append(token_chunk)
                    yield sse_event("token", token_chunk)
            except GenerationError:
                # Fallback: non-streaming generation
                logger.warning("Streaming generation failed, falling back to batch generation")
                collected_tokens = []
                full_text = mlx_generate(
                    model=body.model,
                    prompt=prompt,
                    max_tokens=body.max_tokens,
                    temperature=body.temperature,
                    top_p=body.top_p,
                    seed=body.seed,
                )
                collected_tokens.append(full_text)
                yield sse_event("token", full_text)

            answer_text = "".join(collected_tokens).strip()
            if not answer_text:
                yield sse_event("error", {"message": "LLM returned empty output."})
                return

            # Step 7: Citation validation + repair
            valid, reason = validate_answer_citations(answer_text, len(chunks))

            if not valid:
                repair_prompt = build_mode_repair_prompt(
                    decision.mode,
                    body.question,
                    chunks,
                    answer_text,
                    route_preface=decision.preface,
                )
                try:
                    repaired = mlx_generate(
                        model=body.model,
                        prompt=repair_prompt,
                        max_tokens=body.max_tokens,
                        temperature=body.temperature,
                        top_p=body.top_p,
                        seed=body.seed,
                    )
                    valid, reason = validate_answer_citations(repaired, len(chunks))
                    if valid:
                        answer_text = repaired
                        yield sse_event("repair", repaired)
                except GenerationError as exc:
                    logger.warning("Repair generation failed: %s", exc)

            if not valid and not body.no_refuse:
                yield sse_event("error", {
                    "message": f"Citation validation failed: {reason}",
                    "draft": answer_text,
                })
                return

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            done_payload = AnswerResponse(
                answer=answer_text,
                sources=_chunks_to_response(chunks),
                route=route_info,
                citation_valid=valid,
                citation_reason=reason,
                elapsed_ms=round(elapsed_ms, 1),
            )
            yield sse_event("done", done_payload.model_dump())

        except GenerationError as exc:
            yield sse_event("error", {"message": str(exc)})
        except Exception as exc:
            logger.exception("Unexpected error in answer endpoint")
            yield sse_event("error", {"message": f"Internal error: {exc}"})

    return StreamingResponse(generate(), media_type="text/event-stream")
