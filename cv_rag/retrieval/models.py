from __future__ import annotations

from dataclasses import dataclass, field

from cv_rag.shared.errors import RetrievalError


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    arxiv_id: str
    title: str
    section_title: str
    text: str
    fused_score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    sources: set[str] = field(default_factory=set)


class NoRelevantSourcesError(RetrievalError):
    def __init__(self, candidates: list[RetrievedChunk]) -> None:
        super().__init__("no relevant sources")
        self.candidates = candidates


def format_citation(arxiv_id: str, section_title: str) -> str:
    section = section_title.strip() or "Untitled"
    return f"arXiv:{arxiv_id} §{section}"
