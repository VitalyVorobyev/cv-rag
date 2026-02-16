# Workflow: Feature / Bug Fix / Refactor

All work follows the same flow with varying spec depth.

## Step 1: Specification (Architect — ChatGPT)

1. Read `CLAUDE.md` and `.ai/state/backlog.md`
2. Add a task entry to backlog with status `specifying`
3. Write a task spec using `.ai/templates/task-spec.md`
4. Save to `.ai/state/sessions/YYYY-MM-DD-TASK-NNN-spec.md`
5. If non-obvious architectural choice: write an ADR in `.ai/state/decisions/`
6. Create a handoff note: Architect -> Implementer
7. Update backlog status to `implementing`

For **bug fixes**: spec can be minimal — describe the bug, how to reproduce, and which module is affected.

For **refactors**: list all files to move/rename, define checkpoints where tests must pass.

## Step 2: Implementation (Implementer — Codex)

1. Create a branch: `git checkout -b feat/TASK-NNN-short-name` (or `fix/` or `refactor/`)
2. Read the task spec and handoff note
3. Implement in dependency order:
   a. `shared/` changes (errors, settings, types) first
   b. Domain service logic next
   c. Interface layer (CLI/API) last
4. After each file: `uv run ruff check cv_rag/`
5. After all files: `uv run pytest -q` (no regressions)
6. Create handoff note: Implementer -> Reviewer
7. Update backlog status to `reviewing`

For **bug fixes**: write a failing test first, then fix the code.

## Step 3: Review + Test (Reviewer — Codex)

1. Read the Implementer's handoff and the original spec
2. Run through the review checklist (see `.ai/roles/reviewer.md`)
3. Check dependency direction rules
4. If issues found: handoff back to Implementer, keep status `implementing`
5. If clean: write tests in `tests/test_<module>.py`
6. Run `uv run pytest -q` — all must pass
7. Run `uv run ruff check tests/`
8. Create final handoff note, update backlog status to `done`

## Step 4: Merge (Human)

1. Final check: `uv run pytest -q && uv run ruff check cv_rag/ tests/`
2. Workflow check: `uv run python .ai/scripts/validate_workflow.py`
3. Merge branch to main
4. If architecture docs need updating, Integrator updates `CLAUDE.md` / `AGENTS.md`

## Hard Gate

`AI Workflow Gate` CI blocks merge when backlog/spec/handoff artifacts are inconsistent.
