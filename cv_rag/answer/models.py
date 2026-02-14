from __future__ import annotations

from dataclasses import dataclass, field

from cv_rag.answer.routing import RouteDecision
from cv_rag.retrieval.models import RetrievedChunk


@dataclass(slots=True)
class AnswerRunRequest:
    question: str
    model: str
    mode: str = "auto"
    router_model: str | None = None
    router_strategy: str = "hybrid"
    router_top_k: int = 12
    k: int | None = None
    max_per_doc: int | None = None
    section_boost: float = 0.05
    max_tokens: int = 600
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None = None
    no_refuse: bool = False


@dataclass(slots=True)
class AnswerRunResult:
    answer: str
    sources: list[RetrievedChunk]
    route: RouteDecision
    citation_valid: bool
    citation_reason: str
    warnings: list[str] = field(default_factory=list)
    maybe_relevant: list[RetrievedChunk] = field(default_factory=list)


@dataclass(slots=True)
class AnswerEvent:
    event: str
    data: object
