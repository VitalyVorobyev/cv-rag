# AI Agent Development Workflow

Structured workflow for AI-assisted development of cv-rag.

## Agent Team

| Role | Tool | Responsibility |
|------|------|---------------|
| **Architect** | ChatGPT | Design, specs, ADRs, interface contracts |
| **Implementer** | Codex (VS Code) | Write production code |
| **Reviewer** | Codex (VS Code) | Code review + write tests |
| **Integrator** | Claude Code | Cross-cutting changes, docs, large refactors |

## Quick Start

1. **Architect** creates a task spec in `.ai/state/sessions/` and adds it to `state/backlog.md`
2. **Implementer** picks up the task, reads the handoff, codes on a feature branch
3. **Reviewer** checks the code, writes tests, approves or sends back
4. **Human** merges to main

## Directory Layout

```
roles/           System prompts for each agent role
workflows/       Step-by-step process definitions
state/
  backlog.md     Living task board
  decisions/     Architecture Decision Records
  sessions/      Handoff notes between agents
templates/       Reusable templates for specs and handoffs
```

## Key Files

- Start here: [workflows/feature.md](workflows/feature.md)
- Handoff rules: [workflows/handoff.md](workflows/handoff.md)
- Current tasks: [state/backlog.md](state/backlog.md)
