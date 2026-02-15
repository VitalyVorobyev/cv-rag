# Handoff: Implementer -> Reviewer

**Task**: TASK-010 — Add MLX runner tests  
**Date**: 2026-02-15

## Files Changed

- `tests/test_mlx_runner.py` — created — unit tests for `mlx_generate`, `mlx_generate_stream`, and `sse_event`

## Summary

Implemented focused offline unit tests for `cv_rag/answer/mlx_runner.py` by monkeypatching subprocess entry points:

- `mlx_generate`:
  - success path trims output and validates command construction
  - seed flag inclusion (`--seed 123`)
  - missing runtime (`FileNotFoundError` -> `GenerationError`)
  - non-zero exit with detail (`stderr`)
  - non-zero exit without detail (exit code fallback)
  - empty output on success return code
- `mlx_generate_stream`:
  - success streaming preserves yielded chunk order/content
  - missing runtime (`FileNotFoundError` -> `GenerationError`)
  - non-zero exit after streaming raises `GenerationError` after yielded chunks
- `sse_event`:
  - multiline string payload formatting (`data:` per line + terminating blank line)
  - JSON payload formatting for non-string data

## Deviations / Open Questions

- No deviations from the task spec.
- No open questions.

## Verification

- `uv run ruff check cv_rag/` (after each file edit): passed
- `uv run ruff check tests/test_mlx_runner.py`: passed
- `uv run pytest -q`: passed (`152 passed`)
