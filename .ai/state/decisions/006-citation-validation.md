# ADR-006: Citation validation and repair loop

**Date**: 2026-02-15
**Status**: accepted

## Context

LLM-generated answers may fabricate claims or fail to cite sources. Need a mechanism to enforce grounding in retrieved sources.

## Decision

All answers must include `[S#]` inline citations referencing retrieved sources. Validation (`validate_answer_citations`) checks:
- Every paragraph has at least one `[S#]` citation
- All citation indices are within range `S1..Sk` (k = source count)
- Minimum citations >= paragraph count

If validation fails, a repair prompt (`build_mode_repair_prompt`) appends the draft answer and asks the LLM to add citations and remove unsupported claims. If repair also fails, the answer is refused (unless `--no-refuse` overrides).

For COMPARE mode, cross-document enforcement requires >= 2 sources from each of the top 2 papers.

## Consequences

- Positive: Every claim is traceable to a source; catches hallucination
- Negative: Adds one generation call in worst case; some valid-but-unstructured answers may be refused
- Neutral: `--no-refuse` provides an escape hatch for manual review

## Affected Files

- `cv_rag/answer/citations.py` — `validate_answer_citations()`, `enforce_cross_doc_support()`
- `cv_rag/answer/prompts.py` — citation rules in prompt, `build_mode_repair_prompt()`
- `cv_rag/answer/service.py` — validation + repair loop in `run()` and `stream()`
