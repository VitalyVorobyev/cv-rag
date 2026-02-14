from __future__ import annotations

import re

from cv_rag.retrieval.models import RetrievedChunk

QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
QUESTION_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
SECTION_PRIORITY_RE = re.compile(
    r"(method|approach|implementation|supervision|loss|training|optimal matching|experiments?|ablation|results?)",
    re.IGNORECASE,
)
ENTITY_TOKEN_WHITELIST = {"loftr", "superglue"}
COMMON_QUERY_TERMS = {
    "about",
    "against",
    "between",
    "compare",
    "difference",
    "differences",
    "does",
    "each",
    "explain",
    "fails",
    "from",
    "idea",
    "into",
    "key",
    "method",
    "objective",
    "paper",
    "show",
    "shows",
    "summarize",
    "their",
    "them",
    "these",
    "this",
    "training",
    "when",
    "with",
}


def _is_camel_caseish(token: str) -> bool:
    return any(char.isupper() for char in token) and any(char.islower() for char in token)


def extract_entity_like_tokens(question: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in QUESTION_ALNUM_RE.finditer(question):
        raw = match.group(0)
        lower = raw.lower()
        include = False
        if len(lower) >= 5:
            include = True
        if any(char.isdigit() for char in raw):
            include = True
        if _is_camel_caseish(raw):
            include = True
        if lower in ENTITY_TOKEN_WHITELIST:
            include = True
        if not include:
            continue
        if lower in seen:
            continue
        seen.add(lower)
        out.append(lower)
    return out


def filter_chunks_by_entity_tokens(chunks: list[RetrievedChunk], entity_tokens: list[str]) -> list[RetrievedChunk]:
    if not entity_tokens:
        return chunks
    token_patterns = [re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE) for token in entity_tokens]
    filtered: list[RetrievedChunk] = []
    for chunk in chunks:
        haystack = f"{chunk.title}\n{chunk.section_title}\n{chunk.text}"
        if any(pattern.search(haystack) for pattern in token_patterns):
            filtered.append(chunk)
    return filtered


def matches_priority_section(chunk: RetrievedChunk) -> bool:
    haystack = f"{chunk.title}\n{chunk.section_title}"
    return SECTION_PRIORITY_RE.search(haystack) is not None


def extract_rare_query_terms(query: str) -> list[str]:
    terms = QUERY_TOKEN_RE.findall(query.casefold())
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term in COMMON_QUERY_TERMS:
            continue
        if len(term) < 5 and not any(char.isdigit() for char in term):
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
    return out


def is_irrelevant_result(
    query: str,
    candidates: list[RetrievedChunk],
    vector_score_threshold: float,
) -> bool:
    if not candidates:
        return True

    rare_terms = extract_rare_query_terms(query)
    if not rare_terms:
        return False

    overlap_found = False
    for chunk in candidates:
        haystack = f"{chunk.title}\n{chunk.text}".casefold()
        if any(term in haystack for term in rare_terms):
            overlap_found = True
            break

    max_vector = max(
        (chunk.vector_score for chunk in candidates if chunk.vector_score is not None),
        default=-1.0,
    )
    return (not overlap_found) and (max_vector < vector_score_threshold)
