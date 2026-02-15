# Task Spec: TASK-NNN — Title

**Author**: Architect
**Date**: YYYY-MM-DD
**Priority**: high | medium | low
**Effort**: small (1-2 files) | medium (3-5 files) | large (6+ files)

## Goal

One sentence describing the user-visible outcome.

## Background

Why this is needed. Link to ADR if one exists.

## Interface Contract

```python
# New/modified public APIs with full type signatures

def new_function(param: str, *, option: int = 10) -> Result:
    """One-line docstring."""
    ...
```

## File Plan

| Action | File | Description |
|--------|------|-------------|
| create | `cv_rag/module/new_file.py` | What it contains |
| modify | `cv_rag/module/existing.py` | What to change |

## Error Handling

- What errors can occur and how to handle them
- Which `CvRagError` subclass to use

## Testing Notes

- Key test cases the Reviewer should cover
- Edge cases to watch for

## Out of Scope

- What this task does NOT include
