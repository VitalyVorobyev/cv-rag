from __future__ import annotations

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    vector_k: int = 12
    keyword_k: int = 12
    max_per_doc: int = 4
    section_boost: float = 0.0


class ChunkResponse(BaseModel):
    chunk_id: str
    arxiv_id: str
    title: str
    section_title: str
    text: str
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    sources: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    chunks: list[ChunkResponse]
    query: str
    elapsed_ms: float


class AnswerRequest(BaseModel):
    question: str
    model: str
    mode: str = "auto"
    router_strategy: str = "rules"
    max_tokens: int = 600
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = None
    k: int | None = None
    max_per_doc: int | None = None
    section_boost: float = 0.05
    no_refuse: bool = False


class RouteInfo(BaseModel):
    mode: str
    targets: list[str]
    k: int
    max_per_doc: int
    confidence: float
    notes: str
    preface: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    sources: list[ChunkResponse]
    route: RouteInfo
    citation_valid: bool
    citation_reason: str
    elapsed_ms: float


class PaperSummary(BaseModel):
    arxiv_id: str
    title: str
    summary: str | None = None
    published: str | None = None
    updated: str | None = None
    authors: list[str] = Field(default_factory=list)
    pdf_url: str | None = None
    abs_url: str | None = None
    chunk_count: int = 0
    tier: int | None = None
    citation_count: int | None = None
    venue: str | None = None


class PaperListResponse(BaseModel):
    papers: list[PaperSummary]
    total: int
    offset: int
    limit: int


class PaperDetailResponse(BaseModel):
    paper: PaperSummary
    chunks: list[ChunkResponse]


class StatsResponse(BaseModel):
    papers_count: int
    chunks_count: int
    chunk_docs_count: int
    metrics_count: int
    papers_without_metrics: int
    pdf_files: int
    tei_files: int
    tier_distribution: dict[str, int]
    top_venues: list[dict[str, object]]


class ServiceHealth(BaseModel):
    service: str
    status: str
    detail: str


class HealthResponse(BaseModel):
    services: list[ServiceHealth]
