from __future__ import annotations

from typer.testing import CliRunner

import cv_rag.interfaces.cli.app as cli_module


def test_seed_awesome_root_command_is_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "seed-awesome",
            "--sources",
            "data/curation/awesome_sources.txt",
        ],
    )

    assert result.exit_code == 1
    assert "Command removed: cv-rag seed-awesome" in result.output
    assert "cv-rag corpus discover-awesome" in result.output


def test_resolve_dois_command_is_removed() -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_module.app,
        [
            "resolve-dois",
            "--dois",
            "data/curation/tierA_dois.txt",
        ],
    )

    assert result.exit_code == 1
    assert "Command removed: cv-rag resolve-dois" in result.output
    assert "cv-rag corpus resolve-openalex" in result.output
