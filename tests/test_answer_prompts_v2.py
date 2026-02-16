from __future__ import annotations

from cv_rag.answer.prompts import build_prompt, build_repair_prompt
from cv_rag.answer.routing import AnswerMode
from cv_rag.retrieval.models import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id="2104.00680:0",
            arxiv_id="2104.00680",
            title="LoFTR",
            section_title="Method",
            text="Detector-free matching details.",
            fused_score=0.9,
        ),
        RetrievedChunk(
            chunk_id="1911.11763:0",
            arxiv_id="1911.11763",
            title="SuperGlue",
            section_title="Results",
            text="Reported matching precision improvements.",
            fused_score=0.85,
        ),
    ]


def test_build_prompt_v2_decision_mode_contains_policy_and_mode_instructions() -> None:
    prompt = build_prompt(
        AnswerMode.DECISION,
        "Which method should I use for low-texture scenes?",
        _chunks(),
    )

    assert "Prompt policy: v2" in prompt
    assert "Mode: DECISION" in prompt
    assert "Recommend a concrete approach" in prompt
    assert "Every sentence must include one or more inline citations" in prompt


def test_build_prompt_v2_compare_mode_includes_grouped_tradeoff_instruction() -> None:
    prompt = build_prompt(
        AnswerMode.COMPARE,
        "Compare LoFTR and SuperGlue.",
        _chunks(),
    )

    assert "Mode: COMPARE" in prompt
    assert "include concise semicolon-separated grouped tradeoffs/comparisons" in prompt


def test_build_repair_prompt_v2_contains_draft_payload() -> None:
    repair_prompt = build_repair_prompt(
        AnswerMode.EXPLAIN,
        "Explain LoFTR supervision.",
        _chunks(),
        "Draft answer without citations.",
    )

    assert "Draft to rewrite:" in repair_prompt
    assert "Draft answer without citations." in repair_prompt
    assert "Rewrite the draft. Add inline [S#] citations" in repair_prompt
