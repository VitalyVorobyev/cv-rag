from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from cv_rag.interfaces.api.deps import get_retriever
from cv_rag.interfaces.api.schemas import ChunkResponse, SearchRequest, SearchResponse
from cv_rag.retrieval.hybrid import HybridRetriever

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def search_chunks(
    body: SearchRequest,
    retriever: HybridRetriever = Depends(get_retriever),
) -> SearchResponse:
    t0 = time.perf_counter()

    chunks = retriever.retrieve(
        query=body.query,
        top_k=body.top_k,
        vector_k=body.vector_k,
        keyword_k=body.keyword_k,
        max_per_doc=body.max_per_doc,
        section_boost=body.section_boost,
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return SearchResponse(
        chunks=[
            ChunkResponse(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
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
        ],
        query=body.query,
        elapsed_ms=round(elapsed_ms, 1),
    )
