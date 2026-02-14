from __future__ import annotations

import logging
from collections.abc import Generator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from cv_rag.answer.mlx_runner import sse_event
from cv_rag.answer.models import AnswerRunRequest
from cv_rag.answer.service import AnswerService
from cv_rag.interfaces.api.deps import get_app_settings, get_retriever
from cv_rag.interfaces.api.schemas import AnswerRequest, AnswerResponse, ChunkResponse, RouteInfo
from cv_rag.retrieval.hybrid import HybridRetriever
from cv_rag.retrieval.models import RetrievedChunk
from cv_rag.shared.settings import Settings

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


@router.post("/answer")
def answer_question(
    body: AnswerRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    settings: Settings = Depends(get_app_settings),
) -> StreamingResponse:
    request = AnswerRunRequest(
        question=body.question,
        model=body.model,
        mode=body.mode,
        router_strategy=body.router_strategy,
        router_top_k=12,
        max_tokens=body.max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        seed=body.seed,
        k=body.k,
        max_per_doc=body.max_per_doc,
        section_boost=body.section_boost,
        no_refuse=body.no_refuse,
    )
    service = AnswerService(retriever=retriever, settings=settings)

    def generate() -> Generator[str, None, None]:
        for event in service.stream(request):
            if event.event == "sources":
                chunks = event.data if isinstance(event.data, list) else []
                payload = [c.model_dump() for c in _chunks_to_response(chunks)]
                yield sse_event("sources", payload)
                continue

            if event.event == "done":
                data = event.data if isinstance(event.data, dict) else {}
                sources = data.get("sources", [])
                route = data.get("route")
                if isinstance(route, dict):
                    route_info = RouteInfo(**{
                        "mode": str(route.get("mode", "single")),
                        "targets": list(route.get("targets", [])),
                        "k": int(route.get("k", 8)),
                        "max_per_doc": int(route.get("max_per_doc", 4)),
                        "confidence": float(route.get("confidence", 0.5)),
                        "notes": str(route.get("notes", "")),
                        "preface": route.get("preface"),
                    })
                elif hasattr(route, "mode"):
                    route_info = RouteInfo(
                        mode=str(route.mode.value),
                        targets=list(route.targets),
                        k=int(route.k),
                        max_per_doc=int(route.max_per_doc),
                        confidence=float(route.confidence),
                        notes=str(route.notes),
                        preface=route.preface,
                    )
                else:
                    route_info = RouteInfo(
                        mode="single",
                        targets=[],
                        k=8,
                        max_per_doc=4,
                        confidence=0.5,
                        notes="",
                        preface=None,
                    )

                response = AnswerResponse(
                    answer=str(data.get("answer", "")),
                    sources=_chunks_to_response(sources if isinstance(sources, list) else []),
                    route=route_info,
                    citation_valid=bool(data.get("citation_valid", False)),
                    citation_reason=str(data.get("citation_reason", "")),
                    elapsed_ms=float(data.get("elapsed_ms", 0.0)),
                )
                yield sse_event("done", response.model_dump())
                continue

            yield sse_event(event.event, event.data)

    return StreamingResponse(generate(), media_type="text/event-stream")
