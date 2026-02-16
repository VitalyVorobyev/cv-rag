# Handoff Protocol

## What is a Handoff?

A handoff is a markdown file that passes work from one agent to another. It is the communication bridge between agent sessions.

## Rules

1. Every handoff is a file saved to `.ai/state/sessions/`
2. Naming: `YYYY-MM-DD-TASK-NNN-FROM-TO.md` (e.g., `2026-02-15-TASK-003-architect-implementer.md`)
3. The sending agent updates `.ai/state/backlog.md` to reflect the new status
4. The receiving agent MUST read the handoff note before starting work
5. Handoff notes are append-only — do not modify a sent handoff
6. Workflow postmortems use: `YYYY-MM-DD-TASK-NNN-workflow-postmortem.md`

## Handoff Chain

```
Architect -> Implementer -> Reviewer -> Done
                ^              |
                |              v
                +---- (issues found)
```

## Context Loading Order

When an agent starts a session, it reads files in this order:
1. `AGENTS.md` (or `CLAUDE.md` for Claude Code)
2. `.ai/state/backlog.md` — find the current task
3. The latest handoff note for that task
4. Files listed in the handoff note

## Required Handoff Fields

- Task ID and title
- Date
- Branch name
- Summary (what was done)
- Files changed (with brief descriptions)
- Next step (what the receiver should do)
- Verification commands to run

## Enforcement

Backlog/session consistency is CI-gated by:

```bash
uv run python .ai/scripts/validate_workflow.py
```
