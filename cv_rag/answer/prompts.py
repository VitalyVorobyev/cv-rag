from __future__ import annotations

from cv_rag.answer.routing import PROMPT_SPECS, AnswerMode
from cv_rag.retrieval.models import RetrievedChunk


def _render_sources(chunks: list[RetrievedChunk]) -> list[str]:
    lines: list[str] = []
    lines.append("Sources:")
    for idx, chunk in enumerate(chunks, start=1):
        section = chunk.section_title.strip() or "Untitled"
        snippet = " ".join(chunk.text.split())
        lines.append(f"[S{idx}]")
        lines.append(f"arxiv_id: {chunk.arxiv_id}")
        lines.append(f"title: {chunk.title}")
        lines.append(f"section: {section}")
        lines.append(f"text: {snippet}")
        lines.append("")
    return lines


def _mode_instructions(mode: AnswerMode) -> list[str]:
    if mode is AnswerMode.COMPARE:
        return [
            "Mode: COMPARE",
            "Contrast at least two papers explicitly and attribute each claim to relevant sources.",
            "Prioritize differences in objective, architecture, supervision, data regime, and failure modes.",
            "Use grouped points where useful, but keep concise.",
        ]
    if mode is AnswerMode.SURVEY:
        return [
            "Mode: SURVEY",
            "Map the landscape of options from the sources; group approaches by idea family.",
            "Cover tradeoffs and when to choose each option.",
            "Use grouped points where useful, but keep concise.",
        ]
    if mode is AnswerMode.IMPLEMENTATION:
        return [
            "Mode: IMPLEMENTATION",
            "Provide practical implementation steps grounded in sources only.",
            "Include assumptions, pipeline order, and common pitfalls mentioned by papers.",
            "Do not invent code details absent from sources.",
        ]
    if mode is AnswerMode.EVIDENCE:
        return [
            "Mode: EVIDENCE",
            "Focus on what papers claim and what evidence supports each claim.",
            "Prioritize results, ablations, metrics, and limitations in the provided excerpts.",
            "Call out missing evidence explicitly when needed.",
        ]
    return [
        "Mode: SINGLE_PAPER",
        "Explain the core method or paper clearly and compactly.",
        "Prioritize mechanism, assumptions, and concrete takeaways grounded in sources.",
    ]


def build_prompt(
    mode: AnswerMode,
    question: str,
    chunks: list[RetrievedChunk],
    *,
    route_preface: str | None = None,
) -> str:
    spec = PROMPT_SPECS[mode]

    lines: list[str] = []
    lines.append("You are a careful computer vision research assistant.")
    lines.append(spec.instruction_focus)
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
    lines.extend(_render_sources(chunks))
    lines.extend(_mode_instructions(mode))
    lines.append("")
    lines.append("Rules:")
    lines.append("1. Only use information supported by the sources.")
    lines.append("2. Every sentence must include one or more inline citations like [S3] or [S3][S7].")
    lines.append("3. Every paragraph must include at least one inline citation [S#].")
    lines.append("4. The first paragraph must end with at least one citation.")
    lines.append("5. If sources are insufficient, state what is missing and ask one clarifying question.")
    lines.append("6. Never cite a source outside S1..Sk. Never cite unrelated sources.")
    lines.append("7. Do not add a separate 'Citations' footer.")
    lines.append("8. Keep output compact and structured.")
    lines.append("9. Do not fabricate details, metrics, or citations.")
    lines.append("10. Use paper-accurate terminology.")
    if route_preface:
        lines.append(
            "11. Start P1 with a short routing note consistent with this sentence, then continue with content: "
            f"\"{route_preface}\""
        )
    lines.append("")
    lines.append("Output format (MANDATORY):")
    lines.append("Write EXACTLY 4 paragraphs labeled P1..P4, no headings.")
    lines.append("Each paragraph must have 2 sentences.")
    lines.append("EVERY sentence must end with one or more citations like [S1] or [S1][S2].")
    if mode in {AnswerMode.COMPARE, AnswerMode.SURVEY}:
        lines.append("For P2 or P3, include grouped comparisons/options in concise semicolon-separated clauses.")
    lines.append("Do not output any text before P1.")
    lines.append("")
    lines.append("Template (follow exactly):")
    lines.append("P1: <2 sentences> [S#]")
    lines.append("P2: <2 sentences> [S#]")
    lines.append("P3: <2 sentences> [S#]")
    lines.append("P4: <2 sentences> [S#]")
    lines.append("Answer:")
    return "\n".join(lines)


def build_repair_prompt(
    mode: AnswerMode,
    question: str,
    chunks: list[RetrievedChunk],
    draft_text: str,
    *,
    route_preface: str | None = None,
) -> str:
    base_prompt = build_prompt(mode, question, chunks, route_preface=route_preface)
    return (
        f"{base_prompt}\n\n"
        "Draft to rewrite:\n"
        f"{draft_text}\n\n"
        "Rewrite the draft. Add inline [S#] citations to every paragraph and every non-trivial claim. "
        "Remove any claim not supported by sources."
    )
