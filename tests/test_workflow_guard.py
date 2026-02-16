from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_validator_module(repo_root: Path):
    module_path = repo_root / ".ai" / "scripts" / "validate_workflow.py"
    spec = importlib.util.spec_from_file_location("validate_workflow", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_validate_repo_passes_for_valid_workflow(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".ai" / "state" / "backlog.md",
        "\n".join(
            [
                "# Task Board",
                "",
                "## In Progress",
                "",
                "### [TASK-013] Multi-Source Corpus Growth Architecture",
                "- **Status**: reviewing",
                "",
                "## Backlog",
                "",
                "### [TASK-017] Workflow compliance hard CI gate",
                "- **Status**: backlog",
                "",
                "## Done",
                "",
                "### [TASK-012] Add IngestService integration tests",
                "- **Completed**: 2026-02-15",
            ]
        )
        + "\n",
    )

    sessions = repo / ".ai" / "state" / "sessions"
    _write(sessions / "2026-02-15-TASK-012-spec.md", "# spec\n")
    _write(sessions / "2026-02-15-TASK-012-architect-implementer.md", "# handoff\n")
    _write(sessions / "2026-02-15-TASK-012-implementer-reviewer.md", "# handoff\n")
    _write(sessions / "2026-02-15-TASK-012-reviewer-final.md", "# handoff\n")

    module = _load_validator_module(Path.cwd())
    issues = module.validate_repo(repo)

    assert issues == []


def test_validate_repo_reports_spec_not_in_backlog(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".ai" / "state" / "backlog.md",
        "# Task Board\n\n## In Progress\n\n(none)\n\n## Backlog\n\n(none)\n\n## Done\n\n(none)\n",
    )
    _write(repo / ".ai" / "state" / "sessions" / "2026-02-15-TASK-099-spec.md", "# spec\n")

    module = _load_validator_module(Path.cwd())
    issues = module.validate_repo(repo)

    codes = {issue.code for issue in issues}
    assert "spec_not_in_backlog" in codes


def test_validate_repo_reports_missing_done_handoff_chain(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".ai" / "state" / "backlog.md",
        "\n".join(
            [
                "# Task Board",
                "",
                "## Done",
                "",
                "### [TASK-012] Add IngestService integration tests",
            ]
        )
        + "\n",
    )
    _write(repo / ".ai" / "state" / "sessions" / "2026-02-15-TASK-012-spec.md", "# spec\n")

    module = _load_validator_module(Path.cwd())
    issues = module.validate_repo(repo)

    handoff_issues = [issue for issue in issues if issue.code == "handoff_missing"]
    assert len(handoff_issues) == 3


def test_validate_repo_reports_invalid_status_and_filename(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(
        repo / ".ai" / "state" / "backlog.md",
        "\n".join(
            [
                "# Task Board",
                "",
                "## Backlog",
                "",
                "### [TASK-017] Workflow compliance hard CI gate",
                "- **Status**: waiting",
            ]
        )
        + "\n",
    )
    _write(repo / ".ai" / "state" / "sessions" / "bad-name.md", "x\n")

    module = _load_validator_module(Path.cwd())
    issues = module.validate_repo(repo)

    codes = {issue.code for issue in issues}
    assert "backlog_status_invalid" in codes
    assert "session_filename_invalid" in codes
