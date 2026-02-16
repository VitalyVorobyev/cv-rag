from __future__ import annotations

from cv_rag.answer.routing import AnswerMode, route, rule_router
from cv_rag.retrieval.models import RetrievedChunk


def _chunk(arxiv_id: str, idx: int, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{arxiv_id}:{idx}",
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        section_title="Method",
        text="Details",
        fused_score=score,
    )


def test_rule_router_maps_decision_cue_to_decision_mode() -> None:
    decision, confidence = rule_router("Which approach should I use for correspondence matching?", [])

    assert decision.mode is AnswerMode.DECISION
    assert confidence >= 0.85
    assert "cue_decision" in decision.reason_codes


def test_route_downgrades_decision_to_survey_when_doc_coverage_is_low() -> None:
    prelim_chunks = [
        _chunk("2104.00680", 0, 0.95),
        _chunk("1911.11763", 0, 0.92),
    ]

    decision = route(
        question="Which one should I pick for matching?",
        prelim_chunks=prelim_chunks,
        model_id="unused",
        strategy="rules",
        enable_decision_mode=True,
    )

    assert decision.mode is AnswerMode.SURVEY
    assert "decision_requires_three_docs" in decision.reason_codes


def test_route_hybrid_uses_llm_when_rule_confidence_below_threshold() -> None:
    prelim_chunks = [
        _chunk("2104.00680", 0, 0.9),
        _chunk("1911.11763", 0, 0.85),
        _chunk("2301.00001", 0, 0.82),
    ]

    def fake_llm_generate(**kwargs: object) -> str:
        _ = kwargs
        return '{"mode":"evidence","targets":[],"k":10,"max_per_doc":4,"require_cross_doc":false,"notes":"llm pick"}'

    decision = route(
        question="Tell me more about performance details.",
        prelim_chunks=prelim_chunks,
        model_id="unused",
        strategy="hybrid",
        llm_generate=fake_llm_generate,
        router_min_confidence=0.95,
    )

    assert decision.mode is AnswerMode.EVIDENCE
    assert "llm_router" in decision.reason_codes
    assert decision.policy_version == "v2"
