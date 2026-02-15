# ADR-001: Domain-driven package structure

**Date**: 2026-02-15
**Status**: accepted

## Context

The codebase grew from a single-module script into a multi-feature system (ingest, retrieval, answer, curation, seeding). Organizing by layer (models/, services/, utils/) would scatter related code across directories. Need a structure that makes dependencies explicit and allows parallel work on independent features.

## Decision

Organize `cv_rag/` by domain: `ingest/`, `retrieval/`, `answer/`, `curation/`, `seeding/`, `storage/`. Cross-cutting concerns go in `shared/`. CLI and API are thin wrappers in `interfaces/`. Runtime construction lives in `app/bootstrap.py`.

Dependency direction is strictly enforced:
```
shared/ <-- storage/ <-- {ingest, retrieval, curation, seeding}
                         retrieval/ <-- answer/
interfaces/ --> (any domain package)
app/ --> (any domain package)
```

Domain packages must NOT import from `interfaces/` or `app/`.

## Consequences

- Positive: Clear ownership per package, easy to reason about dependencies, parallelizable dev
- Negative: More directories and `__init__.py` files; cross-package features require coordination
- Neutral: CLI/API handlers are intentionally thin — all logic in domain services

## Affected Files

- `cv_rag/` — full restructure (commit d6d1850)
- `cv_rag/interfaces/cli/commands/` — thin wrappers calling domain services
- `cv_rag/interfaces/api/routers/` — thin wrappers calling domain services
