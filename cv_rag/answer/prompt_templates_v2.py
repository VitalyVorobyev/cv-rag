from __future__ import annotations

from cv_rag.answer.routing import AnswerMode

MODE_INSTRUCTIONS_V2: dict[AnswerMode, list[str]] = {
    AnswerMode.EXPLAIN: [
        "Mode: EXPLAIN",
        "Explain one method or paper mechanism clearly and compactly.",
        "Prioritize mechanism, assumptions, and concrete takeaways grounded in sources.",
    ],
    AnswerMode.COMPARE: [
        "Mode: COMPARE",
        "Contrast at least two papers explicitly and attribute each claim to relevant sources.",
        "Prioritize differences in objective, architecture, supervision, data regime, and failure modes.",
    ],
    AnswerMode.SURVEY: [
        "Mode: SURVEY",
        "Map the landscape of approaches and group them by idea family.",
        "Cover tradeoffs and when to choose each option from available evidence.",
    ],
    AnswerMode.IMPLEMENT: [
        "Mode: IMPLEMENT",
        "Provide practical implementation steps grounded only in cited sources.",
        "Include assumptions, pipeline order, and common pitfalls explicitly.",
    ],
    AnswerMode.EVIDENCE: [
        "Mode: EVIDENCE",
        "Focus on claims, metrics, ablations, and limitations supported by sources.",
        "Call out missing evidence explicitly where claims are under-supported.",
    ],
    AnswerMode.DECISION: [
        "Mode: DECISION",
        "Recommend a concrete approach with explicit trade-offs and confidence bounds.",
        "State decision criteria and tie each criterion back to cited evidence.",
    ],
}


def render_mode_instructions(mode: AnswerMode) -> list[str]:
    return MODE_INSTRUCTIONS_V2.get(mode, MODE_INSTRUCTIONS_V2[AnswerMode.EXPLAIN])
