from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from cv_rag.retrieval.models import RetrievedChunk

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|better than|worse than)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(should i|which (?:approach|model|method)|what should i use|recommend|best choice|choose between)\b",
    re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class DistributionFeatures:
    top1_share: float
    top2_share: float
    num_docs: int
    entropy: float


def has_compare_cue(question: str) -> bool:
    return _COMPARE_RE.search(question) is not None


def has_decision_cue(question: str) -> bool:
    return _DECISION_RE.search(question) is not None


def compute_distribution_features(chunks: list[RetrievedChunk]) -> DistributionFeatures:
    if not chunks:
        return DistributionFeatures(top1_share=0.0, top2_share=0.0, num_docs=0, entropy=0.0)

    counts = Counter(chunk.arxiv_id for chunk in chunks)
    total = sum(counts.values())
    sorted_counts = sorted(counts.values(), reverse=True)

    top1 = sorted_counts[0] / total if sorted_counts else 0.0
    top2 = sum(sorted_counts[:2]) / total if sorted_counts else 0.0

    entropy = 0.0
    for count in sorted_counts:
        p = count / total
        entropy -= p * math.log(p, 2)

    return DistributionFeatures(
        top1_share=top1,
        top2_share=top2,
        num_docs=len(counts),
        entropy=entropy,
    )


def top_doc_counts(chunks: list[RetrievedChunk]) -> list[int]:
    counts = Counter(chunk.arxiv_id for chunk in chunks)
    return sorted(counts.values(), reverse=True)
