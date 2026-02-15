from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

TASK_ID_RE = re.compile(r"^TASK-\d{3}[a-z]?$", re.IGNORECASE)
BACKLOG_TASK_RE = re.compile(r"^###\s+\[(TASK-\d{3}[a-z]?)\]", re.IGNORECASE)
STATUS_RE = re.compile(r"^-\s+\*\*Status\*\*:\s*(.+?)\s*$", re.IGNORECASE)
SPEC_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-(TASK-\d{3}[a-z]?)-spec\.md$", re.IGNORECASE)
SESSION_FILE_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}-(TASK-\d{3}[a-z]?)-(spec|architect-implementer|implementer-reviewer|reviewer-final|workflow-postmortem)\.md$",
    re.IGNORECASE,
)

ALLOWED_STATUSES = {
    "specifying",
    "implementing",
    "reviewing",
    "done",
    "backlog",
    "in progress",
}

REQUIRED_HANDOFF_SUFFIXES = (
    "architect-implementer",
    "implementer-reviewer",
    "reviewer-final",
)


@dataclass(slots=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(slots=True)
class BacklogTask:
    task_id: str
    section: str
    status: str | None


def _normalize_task_id(task_id: str) -> str:
    return task_id.strip().upper()


def parse_backlog(backlog_path: Path) -> tuple[dict[str, BacklogTask], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    tasks: dict[str, BacklogTask] = {}

    if not backlog_path.exists():
        return tasks, [ValidationIssue(code="backlog_missing", message=f"Missing backlog: {backlog_path}")]

    section = ""
    current_task: BacklogTask | None = None
    for line in backlog_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line.removeprefix("## ").strip()
            current_task = None
            continue

        match = BACKLOG_TASK_RE.match(line.strip())
        if match:
            task_id = _normalize_task_id(match.group(1))
            if not TASK_ID_RE.match(task_id):
                issues.append(
                    ValidationIssue(
                        code="backlog_task_id_invalid",
                        message=f"Invalid task ID in backlog: {task_id}",
                    )
                )
                continue
            current_task = BacklogTask(task_id=task_id, section=section, status=None)
            tasks[task_id] = current_task
            continue

        if current_task is None:
            continue

        status_match = STATUS_RE.match(line.strip())
        if status_match:
            status_value = status_match.group(1).strip().casefold()
            current_task.status = status_value
            if status_value not in ALLOWED_STATUSES:
                issues.append(
                    ValidationIssue(
                        code="backlog_status_invalid",
                        message=(
                            f"Invalid status '{status_match.group(1).strip()}' for {current_task.task_id}; "
                            f"allowed: {', '.join(sorted(ALLOWED_STATUSES))}"
                        ),
                    )
                )

    for task in tasks.values():
        if task.section in {"In Progress", "Backlog"} and task.status is None:
            issues.append(
                ValidationIssue(
                    code="backlog_status_missing",
                    message=f"Missing **Status** for {task.task_id} in section '{task.section}'",
                )
            )

    return tasks, issues


def find_spec_tasks(sessions_dir: Path) -> tuple[set[str], list[ValidationIssue]]:
    issues: list[ValidationIssue] = []
    spec_tasks: set[str] = set()

    if not sessions_dir.exists():
        return spec_tasks, [ValidationIssue(code="sessions_missing", message=f"Missing sessions dir: {sessions_dir}")]

    for path in sorted(sessions_dir.glob("*.md")):
        name = path.name
        session_match = SESSION_FILE_RE.match(name)
        if not session_match:
            issues.append(
                ValidationIssue(
                    code="session_filename_invalid",
                    message=(
                        f"Invalid session filename '{name}'. Expected "
                        "YYYY-MM-DD-TASK-NNN-(spec|architect-implementer|implementer-reviewer|reviewer-final|workflow-postmortem).md"
                    ),
                )
            )
            continue

        task_id = _normalize_task_id(session_match.group(1))
        if session_match.group(2) == "spec":
            spec_tasks.add(task_id)

    for path in sorted(sessions_dir.glob("*.md")):
        spec_match = SPEC_FILE_RE.match(path.name)
        if spec_match:
            spec_tasks.add(_normalize_task_id(spec_match.group(1)))

    return spec_tasks, issues


def validate_done_handoff_chains(
    sessions_dir: Path,
    backlog_tasks: dict[str, BacklogTask],
    spec_tasks: set[str],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for task_id, task in sorted(backlog_tasks.items()):
        if task.section != "Done":
            continue
        if task_id not in spec_tasks:
            continue

        for suffix in REQUIRED_HANDOFF_SUFFIXES:
            pattern = f"*-{task_id}-{suffix}.md"
            matches = list(sessions_dir.glob(pattern))
            if matches:
                continue
            issues.append(
                ValidationIssue(
                    code="handoff_missing",
                    message=f"Done task {task_id} missing handoff artifact: {suffix}",
                )
            )

    return issues


def validate_repo(root: Path) -> list[ValidationIssue]:
    backlog_path = root / ".ai" / "state" / "backlog.md"
    sessions_dir = root / ".ai" / "state" / "sessions"

    backlog_tasks, backlog_issues = parse_backlog(backlog_path)
    spec_tasks, session_issues = find_spec_tasks(sessions_dir)

    issues: list[ValidationIssue] = [*backlog_issues, *session_issues]

    for task_id in sorted(spec_tasks):
        if task_id not in backlog_tasks:
            issues.append(
                ValidationIssue(
                    code="spec_not_in_backlog",
                    message=f"Spec exists for {task_id} but backlog has no matching task entry",
                )
            )

    issues.extend(validate_done_handoff_chains(sessions_dir, backlog_tasks, spec_tasks))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate AI workflow artifacts and backlog state")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing .ai/state",
    )
    args = parser.parse_args(argv)

    issues = validate_repo(args.root)
    if issues:
        print("Workflow validation failed:")
        for issue in issues:
            print(f"- [{issue.code}] {issue.message}")
        return 1

    print("Workflow validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
