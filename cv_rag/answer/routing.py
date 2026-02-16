from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from cv_rag.answer.mlx_runner import mlx_generate
from cv_rag.answer.router_policy import (
    compute_distribution_features,
    has_compare_cue,
    has_decision_cue,
    top_doc_counts,
)
from cv_rag.retrieval.models import RetrievedChunk


class AnswerMode(StrEnum):
    EXPLAIN = "explain"
    COMPARE = "compare"
    SURVEY = "survey"
    IMPLEMENT = "implement"
    EVIDENCE = "evidence"
    DECISION = "decision"


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
    reason_codes: list[str] = field(default_factory=list)
    policy_version: str = "v2"


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
    AnswerMode.EXPLAIN: _ModeDefaults(
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
        max_per_doc=3,
        require_cross_doc=False,
        section_boost_hint=0.04,
    ),
    AnswerMode.IMPLEMENT: _ModeDefaults(
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
    AnswerMode.DECISION: _ModeDefaults(
        k=12,
        max_per_doc=3,
        require_cross_doc=False,
        section_boost_hint=0.06,
    ),
}

PROMPT_SPECS: dict[AnswerMode, PromptSpec] = {
    AnswerMode.EXPLAIN: PromptSpec(
        mode=AnswerMode.EXPLAIN,
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
    AnswerMode.IMPLEMENT: PromptSpec(
        mode=AnswerMode.IMPLEMENT,
        instruction_focus="Give grounded implementation steps and pitfalls.",
    ),
    AnswerMode.EVIDENCE: PromptSpec(
        mode=AnswerMode.EVIDENCE,
        instruction_focus="Focus on claims, metrics, ablations, and experimental evidence.",
    ),
    AnswerMode.DECISION: PromptSpec(
        mode=AnswerMode.DECISION,
        instruction_focus="Make a recommendation with explicit tradeoff reasoning grounded in evidence.",
    ),
}

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
    aliases = {
        "single": AnswerMode.EXPLAIN,
        "implementation": AnswerMode.IMPLEMENT,
    }
    if normalized in aliases:
        return aliases[normalized]

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
    reason_codes: list[str] | None = None,
    policy_version: str = "v2",
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
        reason_codes=reason_codes or [],
        policy_version=policy_version,
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
    features = compute_distribution_features(prelim_chunks)
    return DocDistribution(
        top1_share=features.top1_share,
        top2_share=features.top2_share,
        num_docs=features.num_docs,
        entropy=features.entropy,
    )


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
    return has_compare_cue(question)


def rule_router(
    question: str,
    retrieval_summary: list[dict[str, object]],
    *,
    enable_decision_mode: bool = True,
) -> tuple[RouteDecision, float]:
    q = question.strip()
    explicit_compare = has_compare_cue(q)
    if explicit_compare:
        targets = _extract_compare_targets(q)
        decision = make_route_decision(
            AnswerMode.COMPARE,
            targets=targets,
            notes="Rule router: explicit comparison cue.",
            confidence=0.98,
            reason_codes=["cue_compare"],
        )
        return decision, decision.confidence

    if _IMPLEMENT_RE.search(q):
        decision = make_route_decision(
            AnswerMode.IMPLEMENT,
            notes="Rule router: implementation/how-to cue.",
            confidence=0.94,
            reason_codes=["cue_implement"],
        )
        return decision, decision.confidence

    if _EVIDENCE_RE.search(q):
        decision = make_route_decision(
            AnswerMode.EVIDENCE,
            notes="Rule router: evidence/results cue.",
            confidence=0.92,
            reason_codes=["cue_evidence"],
        )
        return decision, decision.confidence

    if enable_decision_mode and has_decision_cue(q):
        decision = make_route_decision(
            AnswerMode.DECISION,
            notes="Rule router: explicit decision/recommendation cue.",
            confidence=0.9,
            reason_codes=["cue_decision"],
        )
        return decision, decision.confidence

    if _SURVEY_RE.search(q):
        decision = make_route_decision(
            AnswerMode.SURVEY,
            notes="Rule router: survey/options cue.",
            confidence=0.9,
            reason_codes=["cue_survey"],
        )
        return decision, decision.confidence

    summary_doc_count = len(retrieval_summary)
    if summary_doc_count >= 4:
        decision = make_route_decision(
            AnswerMode.SURVEY,
            notes="Rule router: broad question with many prelim docs.",
            confidence=0.62,
            reason_codes=["summary_broad"],
        )
        return decision, decision.confidence

    decision = make_route_decision(
        AnswerMode.EXPLAIN,
        notes="Rule router: default explain mode.",
        confidence=0.86,
        reason_codes=["default_explain"],
    )
    return decision, decision.confidence


class _RouterResponse(BaseModel):
    mode: Literal["single", "explain", "compare", "survey", "implement", "evidence", "decision"]
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
    enable_decision_mode: bool,
) -> str:
    mode_values = ["explain", "compare", "survey", "implement", "evidence"]
    if enable_decision_mode:
        mode_values.append("decision")

    schema = {
        "mode": "|".join(mode_values),
        "targets": ["string"],
        "k": "int (8-16)",
        "max_per_doc": "int (2-8)",
        "require_cross_doc": "bool (true only for compare)",
        "notes": "short rationale",
    }

    lines: list[str] = []
    lines.append("You are a routing classifier for a CV paper RAG assistant.")
    lines.append(f"Choose exactly one answer mode from: {', '.join(mode_values)}.")
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
        reason_codes=["llm_router"],
        policy_version="v2",
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
    enable_decision_mode: bool,
    llm_generate: Callable[..., str] = mlx_generate,
) -> RouteDecision:
    prompt = _build_router_prompt(
        question=question,
        retrieval_summary=retrieval_summary,
        strict_json_only=False,
        enable_decision_mode=enable_decision_mode,
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
            enable_decision_mode=enable_decision_mode,
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
            fallback, _ = rule_router(
                question,
                retrieval_summary,
                enable_decision_mode=enable_decision_mode,
            )
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
                reason_codes=[*fallback.reason_codes, "llm_fallback"],
                policy_version="v2",
            )


def _has_compare_coverage(chunks: list[RetrievedChunk]) -> bool:
    counts = top_doc_counts(chunks)
    return len(counts) >= 2 and counts[0] >= 2 and counts[1] >= 2


def _post_check_decision(
    *,
    question: str,
    prelim_chunks: list[RetrievedChunk],
    decision: RouteDecision,
    enable_decision_mode: bool,
) -> RouteDecision:
    distribution = compute_distribution_features(prelim_chunks)
    explicit_compare = has_explicit_compare_cue(question)
    has_compare_support = _has_compare_coverage(prelim_chunks)

    if decision.mode is AnswerMode.COMPARE and not has_compare_support:
        downgraded_mode = (
            AnswerMode.SURVEY
            if distribution.num_docs >= 3 and distribution.top1_share <= 0.7
            else AnswerMode.EXPLAIN
        )
        return make_route_decision(
            downgraded_mode,
            targets=decision.targets,
            notes=f"{decision.notes} Downgraded from compare due low cross-paper coverage.",
            confidence=max(0.55, decision.confidence - 0.2),
            preface=(
                "Routing note: cross-paper comparison evidence was sparse, so this answer is downgraded "
                f"to {downgraded_mode.value} mode."
            ),
            reason_codes=[*decision.reason_codes, "compare_insufficient_coverage"],
        )

    if decision.mode is AnswerMode.DECISION and not enable_decision_mode:
        return make_route_decision(
            AnswerMode.SURVEY,
            targets=decision.targets,
            notes=f"{decision.notes} Decision mode disabled by settings; switched to survey.",
            confidence=max(0.6, decision.confidence - 0.1),
            preface="Routing note: decision mode is disabled, so this answer uses survey mode.",
            reason_codes=[*decision.reason_codes, "decision_disabled"],
        )

    if decision.mode is AnswerMode.DECISION and distribution.num_docs < 3:
        return make_route_decision(
            AnswerMode.SURVEY,
            targets=decision.targets,
            notes=f"{decision.notes} Decision mode requires >=3 docs; switched to survey.",
            confidence=max(0.6, decision.confidence - 0.05),
            preface="Routing note: fewer than 3 distinct papers were retrieved, so this answer uses survey mode.",
            reason_codes=[*decision.reason_codes, "decision_requires_three_docs"],
        )

    if (
        decision.mode in {AnswerMode.SURVEY, AnswerMode.DECISION}
        and distribution.top1_share >= 0.82
        and not explicit_compare
        and distribution.num_docs >= 1
    ):
        return make_route_decision(
            AnswerMode.EXPLAIN,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: one paper dominates retrieval.",
            confidence=max(decision.confidence, 0.8),
            reason_codes=[*decision.reason_codes, "dominant_single_doc"],
        )

    if (
        decision.mode is AnswerMode.EXPLAIN
        and distribution.num_docs >= 4
        and distribution.top1_share <= 0.5
        and distribution.top2_share <= 0.82
    ):
        return make_route_decision(
            AnswerMode.SURVEY,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: multiple similarly strong docs; switched to survey.",
            confidence=max(0.6, decision.confidence),
            reason_codes=[*decision.reason_codes, "broad_retrieval_distribution"],
        )

    if decision.mode is AnswerMode.SURVEY and distribution.num_docs <= 1:
        return make_route_decision(
            AnswerMode.EXPLAIN,
            targets=decision.targets,
            notes=f"{decision.notes} Post-check: not enough distinct docs for survey mode.",
            confidence=max(0.6, decision.confidence),
            preface="Routing note: only one strong paper was retrieved, so this answer uses explain mode.",
            reason_codes=[*decision.reason_codes, "survey_insufficient_docs"],
        )

    if decision.mode is AnswerMode.SURVEY and distribution.num_docs < 3:
        return RouteDecision(
            mode=decision.mode,
            targets=decision.targets,
            k=decision.k,
            max_per_doc=decision.max_per_doc,
            require_cross_doc=decision.require_cross_doc,
            notes=f"{decision.notes} Survey mode warning: fewer than 3 distinct docs in prelim retrieval.",
            confidence=decision.confidence,
            section_boost_hint=decision.section_boost_hint,
            preface=decision.preface,
            reason_codes=[*decision.reason_codes, "survey_low_doc_coverage"],
            policy_version="v2",
        )

    return RouteDecision(
        mode=decision.mode,
        targets=decision.targets,
        k=decision.k,
        max_per_doc=decision.max_per_doc,
        require_cross_doc=decision.require_cross_doc,
        notes=decision.notes,
        confidence=decision.confidence,
        section_boost_hint=decision.section_boost_hint,
        preface=decision.preface,
        reason_codes=decision.reason_codes,
        policy_version="v2",
    )


def route(
    question: str,
    prelim_chunks: list[RetrievedChunk],
    model_id: str,
    strategy: str = "hybrid",
    *,
    llm_generate: Callable[..., str] = mlx_generate,
    router_min_confidence: float = 0.55,
    enable_decision_mode: bool = True,
) -> RouteDecision:
    strategy_enum = RouterStrategy(strategy.strip().casefold())
    retrieval_summary = summarize_prelim_sources(prelim_chunks, max_docs=6)
    rule_decision, rule_confidence = rule_router(
        question,
        retrieval_summary,
        enable_decision_mode=enable_decision_mode,
    )

    if strategy_enum is RouterStrategy.RULES:
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=rule_decision,
            enable_decision_mode=enable_decision_mode,
        )

    if strategy_enum is RouterStrategy.LLM:
        llm_decision = llm_router(
            question,
            retrieval_summary,
            model_id,
            enable_decision_mode=enable_decision_mode,
            llm_generate=llm_generate,
        )
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=llm_decision,
            enable_decision_mode=enable_decision_mode,
        )

    if rule_confidence >= router_min_confidence:
        return _post_check_decision(
            question=question,
            prelim_chunks=prelim_chunks,
            decision=rule_decision,
            enable_decision_mode=enable_decision_mode,
        )

    llm_decision = llm_router(
        question,
        retrieval_summary,
        model_id,
        enable_decision_mode=enable_decision_mode,
        llm_generate=llm_generate,
    )
    return _post_check_decision(
        question=question,
        prelim_chunks=prelim_chunks,
        decision=llm_decision,
        enable_decision_mode=enable_decision_mode,
    )
