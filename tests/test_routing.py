from __future__ import annotations

from cv_rag.answer.routing import AnswerMode, route, rule_router
from cv_rag.retrieval.hybrid import RetrievedChunk


def _chunk(arxiv_id: str, idx: int, score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{arxiv_id}:{idx}",
        arxiv_id=arxiv_id,
        title=f"Paper {arxiv_id}",
        section_title="Method",
        text="Details",
        fused_score=score,
    )


def test_rule_router_compare_question_maps_to_compare() -> None:
    decision, confidence = rule_router("Compare LoFTR vs SuperGlue", [])
    assert decision.mode is AnswerMode.COMPARE
    assert confidence >= 0.9


def test_rule_router_explain_question_maps_to_single() -> None:
    decision, confidence = rule_router("Explain SuperGlue dustbins", [])
    assert decision.mode is AnswerMode.SINGLE_PAPER
    assert confidence >= 0.7


def test_rule_router_options_question_maps_to_survey() -> None:
    decision, confidence = rule_router("What are options for planar homography estimation?", [])
    assert decision.mode is AnswerMode.SURVEY
    assert confidence >= 0.85


def test_rule_router_implement_question_maps_to_implementation() -> None:
    decision, confidence = rule_router("How to implement SuperGlue matching layer?", [])
    assert decision.mode is AnswerMode.IMPLEMENTATION
    assert confidence >= 0.9


def test_rule_router_evidence_question_maps_to_evidence() -> None:
    decision, confidence = rule_router("What does LoFTR report on HPatches?", [])
    assert decision.mode is AnswerMode.EVIDENCE
    assert confidence >= 0.9


def test_route_post_check_prefers_single_for_dominant_one_doc_unless_explicit_compare() -> None:
    dominant_prelim = [
        _chunk("2104.00680", 0, 0.95),
        _chunk("2104.00680", 1, 0.92),
        _chunk("2104.00680", 2, 0.91),
        _chunk("2104.00680", 3, 0.90),
        _chunk("2104.00680", 4, 0.89),
        _chunk("2104.00680", 5, 0.88),
        _chunk("2104.00680", 6, 0.87),
        _chunk("2104.00680", 7, 0.86),
        _chunk("1911.11763", 0, 0.7),
        _chunk("2201.00001", 0, 0.68),
    ]

    single_decision = route(
        question="Explain LoFTR training objective.",
        prelim_chunks=dominant_prelim,
        model_id="unused",
        strategy="rules",
    )
    assert single_decision.mode is AnswerMode.SINGLE_PAPER

    compare_prelim = [*dominant_prelim, _chunk("1911.11763", 1, 0.69)]
    compare_decision = route(
        question="Compare LoFTR vs SuperGlue on matching reliability.",
        prelim_chunks=compare_prelim,
        model_id="unused",
        strategy="rules",
    )
    assert compare_decision.mode is AnswerMode.COMPARE
