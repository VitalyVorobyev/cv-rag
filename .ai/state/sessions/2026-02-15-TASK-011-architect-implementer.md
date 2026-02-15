# Handoff: Architect -> Implementer

**Task**: TASK-011 — Add AnswerService integration tests
**Date**: 2026-02-15
**Branch**: test/TASK-011-answer-service-tests

## Summary

Create offline, deterministic tests for `cv_rag/answer/service.py` (`AnswerService.run()` and `.stream()`). Use injected fakes for the retriever and MLX generation functions to validate the full orchestration: two-phase retrieval (prelim + final), routing, generation, citation validation, repair loop, and streaming event sequencing (including GenerationError fallback).

## Files Changed

- `.ai/state/sessions/2026-02-15-TASK-011-spec.md` — created — task specification and test matrix
- `.ai/state/sessions/2026-02-15-TASK-011-architect-implementer.md` — created — this handoff
- `.ai/state/backlog.md` — modified — moved TASK-011 to In Progress with spec/handoff links

## Open Questions

- None — follow the spec’s minimum test set; add the optional cases only if quick.

## Verification

```bash
uv run ruff check cv_rag/ tests/
uv run pytest -q
```

## Next Step

Implement `tests/test_answer_service.py` per the spec, then create an Implementer → Reviewer handoff in `.ai/state/sessions/`.
