# ADR-005: Service-oriented architecture

**Date**: 2026-02-15
**Status**: accepted

## Context

Business logic was initially embedded in CLI command handlers. This made it impossible to reuse from the API layer and hard to test without invoking Typer.

## Decision

Domain logic lives in service classes (`IngestService`, `AnswerService`, `CurationService`) and the `HybridRetriever`. Dependencies are injected via constructor parameters. CLI and API handlers are thin wrappers that construct services and call methods.

`AppRuntime` in `app/bootstrap.py` builds the shared dependency graph: settings → stores → retriever. Services accept these as constructor args.

State is immutable within a request. No global singletons except the settings cache (`@lru_cache`).

## Consequences

- Positive: Same logic serves CLI and API; easy to test with monkeypatched deps; clear interfaces
- Negative: More boilerplate in constructors; CLI/API handlers have verbose dependency wiring
- Neutral: Settings cache is acceptable since config doesn't change at runtime

## Affected Files

- `cv_rag/app/bootstrap.py` — `AppRuntime`, `build_retriever_runtime()`
- `cv_rag/ingest/service.py` — `IngestService`
- `cv_rag/answer/service.py` — `AnswerService`
- `cv_rag/curation/service.py` — `CurationService`
- `cv_rag/interfaces/cli/commands/` — thin wrappers
- `cv_rag/interfaces/api/deps.py` — FastAPI dependency injection
