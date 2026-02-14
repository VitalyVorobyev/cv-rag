from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from cv_rag.llm import mlx_generate
from cv_rag.retrieve import RetrievedChunk


class AnswerMode(StrEnum):
    SINGLE_PAPER = "single"
    COMPARE = "compare"
    SURVEY = "survey"
    IMPLEMENTATION = "implement"
    EVIDENCE = "evidence"


class RouterStrategy(StrEnum):
    RULES = "rules"
    LLM = "llm"
    HYBRID = "hybrid"


@dataclass(slots=True, frozen=True)
class PromptSpec:
    mode: AnswerMode
    instruction_focus: str


@dataclass(slots=True, frozen=True)
class RouteDecision:
    mode: AnswerMode
    targets: list[str]
    k: int
    max_per_doc: int
    require_cross_doc: bool
    notes: str
    confidence: float
    section_boost_hint: float
    preface: str | None = None


@dataclass(slots=True, frozen=True)
class DocDistribution:
    top1_share: float
    top2_share: float
    num_docs: int
    entropy: float


@dataclass(slots=True, frozen=True)
class _ModeDefaults:
    k: int
    max_per_doc: int
    require_cross_doc: bool
    section_boost_hint: float


MODE_DEFAULTS: dict[AnswerMode, _ModeDefaults] = {
    AnswerMode.SINGLE_PAPER: _ModeDefaults(
        k=8,
        max_per_doc=6,
        require_cross_doc=False,
        section_boost_hint=0.03,
    ),
    AnswerMode.COMPARE: _ModeDefaults(
        k=12,
        max_per_doc=4,
        require_cross_doc=True,
        section_boost_hint=0.05,
    ),
    AnswerMode.SURVEY: _ModeDefaults(
        k=12,
        max_per_doc=2,
        require_cross_doc=False,
        section_boost_hint=0.04,
    ),
    AnswerMode.IMPLEMENTATION: _ModeDefaults(
        k=10,
        max_per_doc=4,
        require_cross_doc=False,
        section_boost_hint=0.09,
    ),
    AnswerMode.EVIDENCE: _ModeDefaults(
        k=10,
        max_per_doc=4,
        require_cross_doc=False,
        section_boost_hint=0.09,
    ),
}

PROMPT_SPECS: dict[AnswerMode, PromptSpec] = {
    AnswerMode.SINGLE_PAPER: PromptSpec(
        mode=AnswerMode.SINGLE_PAPER,
        instruction_focus="Explain one paper/model mechanism clearly and grounded.",
    ),
    AnswerMode.COMPARE: PromptSpec(
        mode=AnswerMode.COMPARE,
        instruction_focus="Contrast papers directly: similarities, differences, tradeoffs.",
    ),
    AnswerMode.SURVEY: PromptSpec(
        mode=AnswerMode.SURVEY,
        instruction_focus="Map the method landscape and group approaches.",
    ),
    AnswerMode.IMPLEMENTATION: PromptSpec(
        mode=AnswerMode.IMPLEMENTATION,
        instruction_focus="Give grounded implementation steps and pitfalls.",
    ),
    AnswerMode.EVIDENCE: PromptSpec(
        mode=AnswerMode.EVIDENCE,
        instruction_focus="Focus on claims, metrics, ablations, and experimental evidence.",
    ),
}

_COMPARE_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|better than|worse than)\b",
    re.IGNORECASE,
)
_SURVEY_RE = re.compile(
    r"\b(options?|methods?|approaches?|survey|landscape|alternatives?|what are .*options)\b",
    re.IGNORECASE,
)
_IMPLEMENT_RE = re.compile(
    r"\b(how to|implement|implementation|pipeline|steps?|pitfalls?|reproduce|training recipe)\b",
    re.IGNORECASE,
)
_EVIDENCE_RE = re.compile(
    r"\b(ablation|ablations|results?|report|reported|claims?|benchmark|metrics?|hpaches)\b",
    re.IGNORECASE,
)
_VS_TARGET_RE = re.compile(
    r"(?P<a>[A-Za-z0-9_.+\-/ ]{2,60}?)\s+(?:vs\.?|versus)\s+(?P<b>[A-Za-z0-9_.+\-/ ]{2,60})",
    re.IGNORECASE,
)
_COMPARE_TARGET_RE = re.compile(
    r"(?:compare|difference between)\s+"
    r"(?P<a>[A-Za-z0-9_.+\-/ ]{2,60}?)\s+"
    r"(?:and|with)\s+"
    r"(?P<b>[A-Za-z0-9_.+\-/ ]{2,60})",
    re.IGNORECASE,
)
_JSON_BLOCK_RE = re.compile(r"\{", re.MULTILINE)


def mode_from_value(value: str) -> AnswerMode:
    normalized = value.strip().casefold()
    for mode in AnswerMode:
        if mode.value == normalized:
            return mode
    raise ValueError(f"Unsupported mode: {value}")


def make_route_decision(
    mode: AnswerMode,
    *,
    targets: list[str] | None = None,
    notes: str = "",
    confidence: float = 0.5,
    preface: str | None = None,
) -> RouteDecision:
    defaults = MODE_DEFAULTS[mode]
    return RouteDecision(
        mode=mode,
        targets=targets or [],
        k=defaults.k,
        max_per_doc=defaults.max_per_doc,
        require_cross_doc=defaults.require_cross_doc,
        notes=notes,
        confidence=confidence,
        section_boost_hint=defaults.section_boost_hint,
        preface=preface,
    )


def summarize_prelim_sources(prelim_chunks: list[RetrievedChunk], max_docs: int = 6) -> list[dict[str, object]]:
    if max_docs <= 0 or not prelim_chunks:
        return []

    grouped: dict[str, dict[str, object]] = {}
    section_counts: dict[str, Counter[str]] = {}
    for chunk in prelim_chunks:
        group = grouped.get(chunk.arxiv_id)
        if group is None:
            group = {
                "arxiv_id": chunk.arxiv_id,
                "title": chunk.title,
                "best_score": chunk.fused_score,
                "count": 0,
            }
            grouped[chunk.arxiv_id] = group
            section_counts[chunk.arxiv_id] = Counter()
        group["count"] = int(group["count"]) + 1
        group["best_score"] = max(float(group["best_score"]), chunk.fused_score)
        section = chunk.section_title.strip() or "Untitled"
        section_counts[chunk.arxiv_id][section] += 1

    ranked = sorted(
        grouped.values(),
        key=lambda item: (-float(item["best_score"]), -int(item["count"]), str(item["arxiv_id"])),
    )
    out: list[dict[str, object]] = []
    for item in ranked[:max_docs]:
        arxiv_id = str(item["arxiv_id"])
        top_sections = [section for section, _ in section_counts[arxiv_id].most_common(3)]
        out.append(
            {
                "arxiv_id": arxiv_id,
                "title": str(item["title"]),
                "best_score": round(float(item["best_score"]), 6),
                "count": int(item["count"]),
                "top_sections": top_sections,
            }
        )
    return out


def doc_distribution(prelim_chunks: list[RetrievedChunk]) -> DocDistribution:
    if not prelim_chunks:
        return DocDistribution(top1_share=0.0, top2_share=0.0, num_docs=0, entropy=0.0)

    counts = Counter(chunk.arxiv_id for chunk in prelim_chunks)
    total = sum(counts.values())
    sorted_counts = sorted(counts.values(), reverse=True)
    top1 = sorted_counts[0] / total if sorted_counts else 0.0
    top2 = sum(sorted_counts[:2]) / total if sorted_counts else 0.0
    entropy = 0.0
    for count in sorted_counts:
        probability = count / total
        entropy -= probability * math.log(probability, 2)
    return DocDistribution(top1_share=top1, top2_share=top2, num_docs=len(counts), entropy=entropy)


def _extract_compare_targets(question: str) -> list[str]:
    for pattern in (_VS_TARGET_RE, _COMPARE_TARGET_RE):
        match = pattern.search(question)
        if match is None:
            continue
        target_a = " ".join(match.group("a").split())
        target_b = " ".join(match.group("b").split())
        return [target_a.strip(" ,.;:"), target_b.strip(" ,.;:")]
    return []


def has_explicit_compare_cue(question: str) -> bool:
    return _COMPARE_RE.search(question) is not None


def rule_router(
    question: str,
    retrieval_summary: list[dict[str, object]],
) -> tuple[RouteDecision, float]:
    q = question.strip()
    explicit_compare = _COMPARE_RE.search(q) is not None
    if explicit_compare:
        targets = _extract_compare_targets(q)
        decision = make_route_decision(
            AnswerMode.COMPARE,
            targets=targets,
            notes="Rule router: explicit comparison cue.",
            confidence=0.98,
        )
        return decision, decision.confidence

    if _IMPLEMENT_RE.search(q):
        decision = make_route_decision(
            AnswerMode.IMPLEMENTATION,
            notes="Rule router: implementation/how-to cue.",
            confidence=0.94,
        )
        return decision, decision.confidence

    if _EVIDENCE_RE.search(q):
        decision = make_route_decision(
            AnswerMode.EVIDENCE,
            notes="Rule router: evidence/results cue.",
            confidence=0.92,
        )
        return decision, decision.confidence

    if _SURVEY_RE.search(q):
        decision = make_route_decision(
            AnswerMode.SURVEY,
            notes="Rule router: survey/options cue.",
            confidence=0.9,
        )
        return decision, decision.confidence

    summary_doc_count = len(retrieval_summary)
    if summary_doc_count >= 4:
        decision = make_route_decision(
            AnswerMode.SURVEY,
            notes="Rule router: broad question with many prelim docs.",
            confidence=0.62,
        )
        return decision, decision.confidence

    decision = make_route_decision(
        AnswerMode.SINGLE_PAPER,
        notes="Rule router: default single-paper explanation.",
        confidence=0.86,
    )
    return decision, decision.confidence


class _RouterResponse(BaseModel):
    mode: Literal["single", "compare", "survey", "implement", "evidence"]
    targets: list[str] = Field(default_factory=list)
    k: int = Field(default=10, ge=8, le=16)
    max_per_doc: int = Field(default=4, ge=2, le=8)
    require_cross_doc: bool = False
    notes: str = Field(default="")


def _build_router_prompt(
    *,
    question: str,
    retrieval_summary: list[dict[str, object]],
    strict_json_only: bool,
) -> str:
    schema = {
        "mode": "single|compare|survey|implement|evidence",
        "targets": ["string"],
        "k": "int (8-16)",
        "max_per_doc": "int (2-8)",
        "require_cross_doc": "bool (true only for compare)",
        "notes": "short rationale",
    }
    lines: list[str] = []
    lines.append("You are a routing classifier for a CV paper RAG assistant.")
    lines.append("Choose exactly one answer mode from: single, compare, survey, implement, evidence.")
    lines.append("Ground your choice in the question and retrieved-source summary.")
    lines.append("Return one JSON object only. No markdown. No prose.")
    if strict_json_only:
        lines.append("Output JSON only.")
    lines.append("")
    lines.append("Output schema:")
    lines.append(json.dumps(schema, ensure_ascii=False))
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Retrieved source summary (top docs):")
    lines.append(json.dumps(retrieval_summary, ensure_ascii=False))
    return "\n".join(lines)


def _extract_first_json_object(text: str) -> str | None:
    start_match = _JSON_BLOCK_RE.search(text)
    if start_match is None:
        return None

    start = start_match.start()
    in_string = False
    escaped = False
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _normalize_router_payload(payload: _RouterResponse) -> RouteDecision:
    mode = mode_from_value(payload.mode)
    defaults = MODE_DEFAULTS[mode]
    require_cross_doc = mode is AnswerMode.COMPARE
    notes = payload.notes.strip() or "LLM router decision."
    return RouteDecision(
        mode=mode,
        targets=[target.strip() for target in payload.targets if target.strip()][:4],
        k=payload.k if payload.k else defaults.k,
        max_per_doc=payload.max_per_doc if payload.max_per_doc else defaults.max_per_doc,
        require_cross_doc=require_cross_doc,
        notes=notes,
        confidence=0.65,
        section_boost_hint=defaults.section_boost_hint,
    )


def _parse_router_json(raw_text: str) -> RouteDecision:
    json_block = _extract_first_json_object(raw_text)
    if json_block is None:
        raise ValueError("No JSON object found in router output.")
    try:
        payload = _RouterResponse.model_validate_json(json_block)
    except ValidationError as exc:
        raise ValueError(f"Router JSON validation failed: {exc}") from exc
    return _normalize_router_payload(payload)


def llm_router(
    question: str,
    retrieval_summary: list[dict[str, object]],
    model_id: str,
    *,
    llm_generate: Callable[..., str] = mlx_generate,
) -> RouteDecision:
    prompt = _build_router_prompt(
        question=question,
        retrieval_summary=retrieval_summary,
        strict_json_only=False,
    )
    try:
        response_text = llm_generate(
            model=model_id,
            prompt=prompt,
            max_tokens=220,
            temperature=0.0,
            top_p=1.0,
            seed=0,
        )
        return _parse_router_json(response_text)
    except Exception:  # noqa: BLE001
        retry_prompt = _build_router_prompt(
            question=question,
            retrieval_summary=retrieval_summary,
            strict_json_only=True,
        )
        try:
            retry_text = llm_generate(
                model=model_id,
                prompt=retry_prompt,
                max_tokens=220,
                temperature=0.0,
                top_p=1.0,
                seed=0,
            )
            return _parse_router_json(retry_text)
        except Exception:  # noqa: BLE001
            fallback, _ = rule_router(question, retrieval_summary)
            return RouteDecision(
                mode=fallback.mode,
                targets=fallback.targets,
                k=fallback.k,
                max_per_doc=fallback.max_per_doc,
                require_cross_doc=fallback.require_cross_doc,
                notes=f"{fallback.notes} LLM router fallback triggered.",
                confidence=fallback.confidence,
                section_boost_hint=fallback.section_boost_hint,
                preface=fallback.preface,
            )


def _top_doc_counts(chunks: list[RetrievedChunk]) -> list[int]:
    counts = Counter(chunk.arxiv_id for chunk in chunks)
    return sorted(counts.values(), reverse=True)


def _has_compare_coverage(chunks: list[RetrievedChunk]) -> bool:
    counts = _top_doc_counts(chunks)
    return len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2


def _post_check_decision(
    *,
    question: str,
    prelim_chunks: list[RetrievedChunk],
    decision: RouteDecision,
) -> RouteDecision:
    distribution = doc_distribution(prelim_chunks)
    explicit_compare = has_explicit_compare_cue(question)
    has_compare_support = _has_compare_coverage(prelim_chunks)

    if decision.mode is AnswerMode.COMPARE and not has_compare_support:
        downgraded_mode = (
            AnswerMode.SURVEY
            if distribution.num_docs >= 3 and distribution.top1_share <= 0.7
            else AnswerMode.SINGLE_PAPER
        )
        downgraded = make_route_decision(
            downgraded_mode,
            targets=decision.targets,
            notes=f"{decision.notes} Downgraded from compare due low cross-paper coverage.",
            confidence=max(0.55, decision.confidence - 0.2),
            preface=(
                "Routing note: cross-paper comparison evidence was sparse, so this answer is downgraded "
                f"to {downgraded_mode.value} mode."
            ),
        )
        return downgraded

    if (
        decision.mode is not AnswerMode.COMPARE
        and distribution.top1_share >= 0.8
        and not explicit_compare
        and distribution.num_docs >= 1
    ):
        return make_route_decision(
            AnswerMode.SINGLE_PAPER,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: one paper dominates retrieval.",
            confidence=max(decision.confidence, 0.8),
        )

    if (
        decision.mode is AnswerMode.SINGLE_PAPER
        and distribution.num_docs >= 3
        and distribution.top1_share <= 0.52
        and distribution.top2_share <= 0.82
    ):
        return make_route_decision(
            AnswerMode.SURVEY,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: multiple similarly strong docs; switched to survey.",
            confidence=max(0.6, decision.confidence),
        )

    if decision.mode is AnswerMode.SURVEY and distribution.num_docs <= 1:
        return make_route_decision(
            AnswerMode.SINGLE_PAPER,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: not enough distinct docs for survey mode.",
            confidence=max(0.6, decision.confidence),
            preface="Routing note: only one strong paper was retrieved, so this answer uses single-paper mode.",
        )

    if decision.mode is AnswerMode.SURVEY and distribution.num_docs < 4:
        return RouteDecision(
            mode=decision.mode,
            targets=decision.targets,
            k=decision.k,
            max_per_doc=decision.max_per_doc,
            require_cross_doc=decision.require_cross_doc,
            notes=f"{decision.notes} Survey mode warning: fewer than 4 distinct docs in prelim retrieval.",
            confidence=decision.confidence,
            section_boost_hint=decision.section_boost_hint,
            preface=decision.preface,
        )

    return decision


def route(
    question: str,
    prelim_chunks: list[RetrievedChunk],
    model_id: str,
    strategy: str = "hybrid",
    *,
    llm_generate: Callable[..., str] = mlx_generate,
) -> RouteDecision:
    strategy_enum = RouterStrategy(strategy.strip().casefold())
    retrieval_summary = summarize_prelim_sources(prelim_chunks, max_docs=6)
    rule_decision, rule_confidence = rule_router(question, retrieval_summary)

    if strategy_enum is RouterStrategy.RULES:
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=rule_decision,
        )

    if strategy_enum is RouterStrategy.LLM:
        llm_decision = llm_router(
            question,
            retrieval_summary,
            model_id,
            llm_generate=llm_generate,
        )
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=llm_decision,
        )

    # Hybrid: trust explicit high-confidence rules, otherwise delegate to LLM.
    if rule_confidence >= 0.85:
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=rule_decision,
        )
    llm_decision = llm_router(
        question,
        retrieval_summary,
        model_id,
        llm_generate=llm_generate,
    )
    return _post_check_decision(
        question=question,
        prelim_chunks=prelim_chunks,
        decision=llm_decision,
    )
