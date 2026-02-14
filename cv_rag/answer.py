from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from cv_rag.retrieve import (
    HybridRetriever,
    RetrievedChunk,
    filter_chunks_by_entity_tokens,
    format_citation,
)

logger = logging.getLogger(__name__)

COMPARISON_QUERY_RE = re.compile(
    r"\b(compare|comparison|versus|vs\.?|difference|different|better than|worse than)\b",
    re.IGNORECASE,
)
CITATION_REF_RE = re.compile(r"\[S(\d+)\]")
PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+")
PREFACE_RE = re.compile(r"^(answer|citations?|sources?)\s*:\s*$", re.IGNORECASE)
HEADING_RE = re.compile(r"^#{1,6}\s+\S")


def is_comparison_question(question: str) -> bool:
    return COMPARISON_QUERY_RE.search(question) is not None


def top_doc_source_counts(chunks: list[RetrievedChunk]) -> list[tuple[str, int]]:
    if not chunks:
        return []
    best_scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk in chunks:
        counts[chunk.arxiv_id] = counts.get(chunk.arxiv_id, 0) + 1
        best_scores[chunk.arxiv_id] = max(best_scores.get(chunk.arxiv_id, float("-inf")), chunk.fused_score)
    top_docs = sorted(best_scores.keys(), key=lambda arxiv_id: best_scores[arxiv_id], reverse=True)[:2]
    return [(arxiv_id, counts[arxiv_id]) for arxiv_id in top_docs]


def _split_paragraphs(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part.strip() for part in PARAGRAPH_SPLIT_RE.split(stripped) if part.strip()]


def validate_answer_citations(answer_text: str, source_count: int) -> tuple[bool, str]:
    paragraphs = _split_paragraphs(answer_text)

    checked = 0
    total_citations = 0

    for paragraph_idx, paragraph in enumerate(paragraphs, start=1):
        p = paragraph.strip()
        if not p:
            continue
        if PREFACE_RE.match(p):
            continue
        if HEADING_RE.match(p):
            continue

        refs = [int(ref) for ref in CITATION_REF_RE.findall(p)]
        if not refs:
            return False, f"Paragraph {paragraph_idx} has no inline [S#] citation."
        if any(ref < 1 or ref > source_count for ref in refs):
            return False, f"Paragraph {paragraph_idx} cites a source outside S1..S{source_count}."
        total_citations += len(refs)
        checked += 1

    if checked == 0:
        return False, "Answer has no content paragraphs."

    required_total = checked
    if total_citations < required_total:
        return False, f"Found {total_citations} citations, need at least {required_total}."

    return True, ""


@dataclass(slots=True)
class CrossDocSupportDecision:
    filtered_chunks: list[RetrievedChunk]
    should_refuse: bool
    warnings: list[str]


def enforce_cross_doc_support(question: str, chunks: list[RetrievedChunk]) -> CrossDocSupportDecision:
    comparison = is_comparison_question(question)
    top_doc_counts = top_doc_source_counts(chunks)
    enough_cross_doc_support = len(top_doc_counts) >= 2 and all(count >= 2 for _, count in top_doc_counts)

    if comparison:
        if enough_cross_doc_support:
            return CrossDocSupportDecision(filtered_chunks=chunks, should_refuse=False, warnings=[])
        warnings = [
            "Warning: need at least 2 sources from each of the top 2 papers"
            " (by score) for robust comparison grounding."
        ]
        if top_doc_counts:
            details = ", ".join(f"{arxiv_id} ({count})" for arxiv_id, count in top_doc_counts)
            warnings.append(f"Current top-paper source counts: {details}")
        return CrossDocSupportDecision(filtered_chunks=chunks, should_refuse=True, warnings=warnings)

    filtered_chunks = chunks
    if len(top_doc_counts) >= 2:
        top_arxiv, top_count = top_doc_counts[0]
        second_arxiv, second_count = top_doc_counts[1]
        if top_count >= 2 and second_count == 1:
            filtered_chunks = [chunk for chunk in chunks if chunk.arxiv_id != second_arxiv]

    return CrossDocSupportDecision(filtered_chunks=filtered_chunks, should_refuse=False, warnings=[])


def build_query_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    lines.append("You are answering a computer vision research question using paper excerpts.")
    lines.append("Use only the provided context. If context is insufficient, say so.")
    lines.append("Cite claims with inline citations in this format: arXiv:ID §Section.")
    lines.append("")
    lines.append(f"Question: {query}")
    lines.append("")
    lines.append("Context:")

    for idx, chunk in enumerate(chunks, start=1):
        citation = format_citation(chunk.arxiv_id, chunk.section_title)
        lines.append(f"[{idx}] {citation}")
        lines.append(chunk.text)
        lines.append("")

    lines.append("Answer:")
    return "\n".join(lines)


def build_strict_answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    lines: list[str] = []
    lines.append("You are a careful computer vision research assistant.")
    lines.append("")
    lines.append("Question:")
    lines.append(question)
    lines.append("")
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

    lines.append("Rules:")
    lines.append("1. Only use information supported by the sources.")
    lines.append("2. Every sentence must include one or more inline citations like [S3] or [S3][S7].")
    lines.append("3. Every paragraph must include at least one inline citation [S#].")
    lines.append("4. The first paragraph must end with at least one citation.")
    lines.append("5. If sources are insufficient, state what is missing and ask one clarifying question.")
    lines.append("6. Prefer comparisons and what each paper claims or shows.")
    lines.append("7. Never cite a source outside S1..Sk. Never cite unrelated sources.")
    lines.append("8. Do not add a separate 'Citations' footer.")
    lines.append("9. Do not fabricate details, metrics, or citations.")
    lines.append("10. Use paper-accurate terminology.")
    lines.append("")
    lines.append("Output format (MANDATORY):")
    lines.append("Write EXACTLY 4 paragraphs labeled P1..P4, no headings, no bullet lists.")
    lines.append("Each paragraph must have 2 sentences.")
    lines.append("EVERY sentence must end with one or more citations like [S1] or [S1][S2].")
    lines.append("Do not output any text before P1.")
    lines.append("")
    lines.append("Template (follow exactly):")
    lines.append("P1: <2 sentences> [S#]")
    lines.append("P2: <2 sentences> [S#]")
    lines.append("P3: <2 sentences> [S#]")
    lines.append("P4: <2 sentences> [S#]")
    lines.append("Answer:")
    return "\n".join(lines)


def build_repair_prompt(question: str, chunks: list[RetrievedChunk], draft_text: str) -> str:
    base_prompt = build_strict_answer_prompt(question, chunks)
    return (
        f"{base_prompt}\n\n"
        "Draft to rewrite:\n"
        f"{draft_text}\n\n"
        "Rewrite the draft. Add inline [S#] citations to every paragraph and every non-trivial claim. "
        "Remove any claim not supported by sources."
    )


def merge_and_cap_chunks(chunks: list[RetrievedChunk], max_per_doc: int) -> list[RetrievedChunk]:
    by_key: dict[tuple[str, str, str], RetrievedChunk] = {}
    for chunk in chunks:
        key = (
            chunk.arxiv_id,
            chunk.section_title.strip().casefold(),
            chunk.chunk_id,
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = RetrievedChunk(
                chunk_id=chunk.chunk_id,
                arxiv_id=chunk.arxiv_id,
                title=chunk.title,
                section_title=chunk.section_title,
                text=chunk.text,
                fused_score=chunk.fused_score,
                vector_score=chunk.vector_score,
                keyword_score=chunk.keyword_score,
                sources=set(chunk.sources),
            )
            continue

        existing.fused_score += chunk.fused_score
        existing.sources.update(chunk.sources)
        if chunk.vector_score is not None:
            existing.vector_score = (
                chunk.vector_score
                if existing.vector_score is None
                else max(existing.vector_score, chunk.vector_score)
            )
        if chunk.keyword_score is not None:
            existing.keyword_score = (
                chunk.keyword_score
                if existing.keyword_score is None
                else min(existing.keyword_score, chunk.keyword_score)
            )

    ranked = sorted(by_key.values(), key=lambda item: item.fused_score, reverse=True)
    return HybridRetriever._apply_doc_quota(ranked, max_per_doc=max_per_doc)


def retrieve_for_answer(
    retriever: HybridRetriever,
    question: str,
    k: int,
    max_per_doc: int,
    section_boost: float,
    entity_tokens: list[str],
) -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    queries = [question]
    per_query_top_k = max(k, 12)
    per_query_branch_k = max(per_query_top_k * 2, 24)
    merged_input: list[RetrievedChunk] = []

    for retrieval_query in queries:
        merged_input.extend(
            retriever.retrieve(
                query=retrieval_query,
                top_k=per_query_top_k,
                vector_k=per_query_branch_k,
                keyword_k=per_query_branch_k,
                require_relevance=False,
                max_per_doc=0,
                section_boost=section_boost,
            )
        )

    merged = merge_and_cap_chunks(merged_input, max_per_doc=max_per_doc)
    maybe_relevant = merged[:3]
    if entity_tokens:
        merged = filter_chunks_by_entity_tokens(merged, entity_tokens)
    return merged[:k], maybe_relevant
