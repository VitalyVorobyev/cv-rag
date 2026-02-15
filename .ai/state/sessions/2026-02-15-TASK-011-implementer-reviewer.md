# Handoff: Implementer -> Reviewer

**Task**: TASK-011 — Add AnswerService integration tests  
**Date**: 2026-02-15

## Files Changed

- `tests/test_answer_service.py` — created — integration-style unit tests for `AnswerService.run()` and `AnswerService.stream()`

## Summary

Implemented deterministic offline tests for `cv_rag/answer/service.py` using injected fakes for retrieval and generation:

- `run()` happy path with auto mode + rules routing, validating successful orchestration and cited result output.
- `run()` citation repair success path, asserting warning emission and second-pass repair generation.
- `run()` failure path when repair still fails citation validation, asserting `CitationValidationError` with reason and draft.
- `stream()` success event sequence (`route` -> `sources` -> `token*` -> `done`) with citation-valid final payload.
- `stream()` `GenerationError` fallback path from streaming generator to non-stream `generate_fn`.
- `stream()` comparison refusal path emitting an `error` event when forced compare lacks cross-document source support.

## Deviations / Open Questions

- No deviations from spec; included one optional coverage case (stream comparison refusal error event).
- No open questions.

## Verification

- `uv run ruff check cv_rag/`: passed
- `uv run ruff check tests/test_answer_service.py`: passed
- `uv run pytest -q`: passed (`158 passed`)
