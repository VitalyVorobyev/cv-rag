# Workflow Postmortem: TASK-012

**Task**: TASK-012 — Add IngestService integration tests
**Date**: 2026-02-15
**Disposition**: task remains done

## Summary

TASK-012 implementation and tests were completed, but backlog state transitions and handoff protocol were not treated as a hard gate. This allowed the work to land without workflow enforcement.

## What Drifted

- Backlog status progression was not used as a strict checkpoint.
- The handoff protocol was treated as advisory, not required.
- There was no CI validation for `.ai/state/backlog.md` to session artifact consistency.

## Corrective Actions

- Add `TASK-017` to implement a hard CI workflow gate.
- Validate that each spec task exists in backlog.
- Validate required handoff chain for done tasks with specs.
- Validate session filename conventions and allowed backlog status values.

## Prevention

No further task should be merged without passing workflow-gate CI checks.
