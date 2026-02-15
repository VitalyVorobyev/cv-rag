from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import cv_rag.interfaces.cli.app as cli_module
from cv_rag.shared.settings import Settings


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def test_seed_visionbib_command_wires_arguments(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    sources = tmp_path / "visionbib_sources.txt"
    sources.write_text(
        "\n".join(
            [
                "https://www.visionbib.com/bibliography/",
                "compute 42 44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "visionbib_out"
    tier_dois = tmp_path / "tierA_dois_visionbib.txt"
    tier_urls = tmp_path / "tierA_urls_visionbib.txt"
    tier_arxiv = tmp_path / "tierA_arxiv_visionbib.txt"

    def fake_run_seed_visionbib_command(
        *,
        settings: Settings,
        console: object,
        sources: Path,
        out_dir: Path,
        tier_a_dois: Path,
        tier_a_urls: Path,
        tier_a_arxiv: Path,
    ) -> None:
        _ = console
        captured["settings"] = settings
        captured["sources"] = sources
        captured["out_dir"] = out_dir
        captured["tier_a_dois"] = tier_a_dois
        captured["tier_a_urls"] = tier_a_urls
        captured["tier_a_arxiv"] = tier_a_arxiv

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "run_seed_visionbib_command", fake_run_seed_visionbib_command)

    result = runner.invoke(
        cli_module.app,
        [
            "seed",
            "visionbib",
            "--sources",
            str(sources),
            "--out-dir",
            str(out_dir),
            "--tierA-dois",
            str(tier_dois),
            "--tierA-urls",
            str(tier_urls),
            "--tierA-arxiv",
            str(tier_arxiv),
        ],
    )

    assert result.exit_code == 0
    assert captured["settings"] == settings
    assert captured["sources"] == sources
    assert captured["out_dir"] == out_dir
    assert captured["tier_a_dois"] == tier_dois
    assert captured["tier_a_urls"] == tier_urls
    assert captured["tier_a_arxiv"] == tier_arxiv


def test_resolve_dois_passes_optional_arxiv_output(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    captured: dict[str, object] = {}

    dois = tmp_path / "tierA_dois.txt"
    out_dir = tmp_path / "curation"
    tier_urls = tmp_path / "tierA_urls_openalex.txt"
    tier_arxiv = tmp_path / "tierA_arxiv_openalex.txt"

    def fake_run_resolve_dois_command(
        *,
        settings: Settings,
        console: object,
        dois: Path,
        out_dir: Path,
        user_agent: str | None,
        api_key_env: str,
        tier_a_urls: Path,
        tier_a_arxiv_from_openalex: Path | None,
        email: str | None,
    ) -> None:
        _ = console
        captured["settings"] = settings
        captured["dois"] = dois
        captured["out_dir"] = out_dir
        captured["user_agent"] = user_agent
        captured["api_key_env"] = api_key_env
        captured["tier_a_urls"] = tier_a_urls
        captured["tier_a_arxiv_from_openalex"] = tier_a_arxiv_from_openalex
        captured["email"] = email

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "run_resolve_dois_command", fake_run_resolve_dois_command)

    result = runner.invoke(
        cli_module.app,
        [
            "resolve-dois",
            "--dois",
            str(dois),
            "--out-dir",
            str(out_dir),
            "--tierA-urls",
            str(tier_urls),
            "--tierA-arxiv-from-openalex",
            str(tier_arxiv),
            "--email",
            "test@example.com",
        ],
    )

    assert result.exit_code == 0
    assert captured["settings"] == settings
    assert captured["dois"] == dois
    assert captured["out_dir"] == out_dir
    assert captured["tier_a_urls"] == tier_urls
    assert captured["tier_a_arxiv_from_openalex"] == tier_arxiv
    assert captured["email"] == "test@example.com"
