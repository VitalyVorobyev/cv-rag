# ADR-003: Multi-mode answer routing with fallback

**Date**: 2026-02-15
**Status**: accepted

## Context

Different question types (explain a paper, compare two papers, survey a field, implementation guide, evidence analysis) benefit from different retrieval parameters and prompt strategies. Need a routing mechanism that balances speed and accuracy.

## Decision

Five answer modes: `SINGLE_PAPER`, `COMPARE`, `SURVEY`, `IMPLEMENTATION`, `EVIDENCE`. Three routing strategies:

1. **Rules** — regex pattern matching on question keywords (fast, high confidence for explicit cues)
2. **LLM** — MLX model classifies into mode via structured JSON output (nuanced, lower confidence)
3. **Hybrid** (default) — trust rules if confidence >= 0.85, else delegate to LLM

Post-routing adjustments (`_post_check_decision`): downgrade COMPARE if insufficient cross-doc sources, upgrade to SURVEY if many similar-strength docs, etc.

Each mode has tuned defaults for `k`, `max_per_doc`, `require_cross_doc`, `section_boost_hint`.

## Consequences

- Positive: Mode-specific retrieval and prompts improve answer quality; rules are fast for obvious cases
- Negative: LLM routing adds latency; hybrid strategy has two retrieval passes (prelim + final)
- Neutral: Post-check adjustments can override the initial decision

## Affected Files

- `cv_rag/answer/routing.py` — `route()`, `rule_router()`, `llm_router()`, `MODE_DEFAULTS`
- `cv_rag/answer/prompts.py` — mode-specific prompt builders
- `cv_rag/answer/service.py` — two-pass retrieval (prelim for routing, final for answer)
