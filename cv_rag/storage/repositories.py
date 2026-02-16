from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReferenceType = Literal["arxiv", "doi", "pdf_url"]
CandidateStatus = Literal["discovered", "resolved", "ready", "ingested", "blocked", "failed"]

SOURCE_WEIGHTS: dict[str, float] = {
    "curated_repo": 1.00,
    "arxiv_feed": 0.90,
    "openalex_resolved": 0.70,
    "scraped_pdf": 0.50,
}

RESOLUTION_WEIGHTS: dict[str, float] = {
    "arxiv_versioned": 0.40,
    "oa_pdf": 0.30,
    "pdf_only": 0.20,
    "metadata_only": 0.00,
}


@dataclass(slots=True, frozen=True)
class ReferenceRecord:
    ref_type: ReferenceType
    normalized_value: str
    source_kind: str
    source_ref: str
    discovered_at_unix: int

    @property
    def doc_id(self) -> str:
        return build_doc_id_from_reference(self.ref_type, self.normalized_value)


@dataclass(slots=True, frozen=True)
class ResolvedReference:
    doc_id: str
    arxiv_id: str | None
    arxiv_id_with_version: str | None
    doi: str | None
    pdf_url: str | None
    resolution_confidence: float
    source_kind: str = "openalex_resolved"
    resolved_at_unix: int | None = None


@dataclass(slots=True, frozen=True)
class IngestCandidate:
    doc_id: str
    status: CandidateStatus
    best_pdf_url: str | None
    priority_score: float
    retry_count: int
    next_retry_unix: int | None


@dataclass(slots=True, frozen=True)
class PaperRecord:
    arxiv_id: str
    arxiv_id_with_version: str
    version: str | None
    title: str
    summary: str
    published: str | None
    updated: str | None
    authors: list[str]
    pdf_url: str
    abs_url: str
    pdf_path: Path | None
    tei_path: Path | None
    doc_id: str | None = None
    provenance_kind: str | None = None
    content_sha256: str | None = None


def build_axv_doc_id(arxiv_id_with_version: str) -> str:
    return f"axv:{arxiv_id_with_version.strip()}"


def build_url_doc_id(pdf_url: str) -> str:
    digest = hashlib.sha256(pdf_url.strip().encode("utf-8")).hexdigest()
    return f"url:{digest}"


def build_doc_id_from_reference(ref_type: ReferenceType, normalized_value: str) -> str:
    value = normalized_value.strip()
    if ref_type == "arxiv":
        return build_axv_doc_id(value)
    if ref_type == "doi":
        return f"doi:{value}"
    return build_url_doc_id(value)


def classify_provenance_kind(source_kind: str) -> str:
    value = source_kind.strip().casefold()
    if value in {
        "curated_repo",
        "awesome_repo",
        "awesome",
        "visionbib",
        "visionbib_page",
    }:
        return "curated"
    if value in {
        "arxiv_feed",
        "arxiv_api",
        "openalex_resolved",
        "semantic_scholar",
        "canonical_api",
    }:
        return "canonical_api"
    return "scraped"


def compute_candidate_priority(
    *,
    source_kind: str,
    resolution_confidence: float,
    age_days: int,
    retry_count: int,
    resolution_kind: str = "metadata_only",
) -> float:
    source_weight = SOURCE_WEIGHTS.get(source_kind, SOURCE_WEIGHTS["scraped_pdf"])
    resolution_weight = RESOLUTION_WEIGHTS.get(
        resolution_kind,
        RESOLUTION_WEIGHTS["metadata_only"],
    )
    freshness_bonus = max(0.0, 0.2 - (max(age_days, 0) / 365.0))
    retry_penalty = 0.15 * max(retry_count, 0)
    confidence_bonus = max(0.0, min(resolution_confidence, 1.0)) * 0.05
    return source_weight + resolution_weight + freshness_bonus + confidence_bonus - retry_penalty
