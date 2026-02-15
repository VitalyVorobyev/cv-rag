# Handoff: [FROM_ROLE] -> [TO_ROLE]

**Task**: TASK-NNN — Brief title
**Date**: YYYY-MM-DD
**Branch**: branch-name

## Summary

One paragraph: what was done and what comes next.

## Files Changed

- `cv_rag/path/file.py` — created | modified — brief description
- `tests/test_file.py` — created — tests for ...

## Open Questions

- (List any ambiguities or deviations from spec)

## Verification

```bash
uv run ruff check cv_rag/ tests/
uv run pytest -q
```

## Next Step

What the receiving role should do with this.
