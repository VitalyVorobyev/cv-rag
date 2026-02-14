from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import cv_rag.interfaces.cli.app as cli_module
from cv_rag.retrieval.hybrid import RetrievedChunk
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


def test_load_eval_cases_applies_defaults(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text("- question: test question\n", encoding="utf-8")

    cases = cli_module._load_eval_cases(suite_path)

    assert len(cases) == 1
    assert cases[0].question == "test question"
    assert cases[0].min_sources == 6
    assert cases[0].must_include_arxiv_ids == []
    assert cases[0].must_include_tokens == []


def test_eval_command_passes_with_matching_sources_and_citations(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        (
            "- question: Explain LoFTR\n"
            "  must_include_arxiv_ids: ['2104.00680']\n"
            "  must_include_tokens: ['loftr']\n"
            "  min_sources: 2\n"
        ),
        encoding="utf-8",
    )

    class DummyQdrantStore:
        def __init__(self, url: str, collection_name: str) -> None:
            self.url = url
            self.collection_name = collection_name

    def fake_retrieve(
        self: object,
        query: str,
        top_k: int = 8,
        vector_k: int = 12,
        keyword_k: int = 12,
        require_relevance: bool = False,
        vector_score_threshold: float = 0.45,
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="2104.00680:0",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Method",
                text="LoFTR uses detector-free matching.",
                fused_score=0.9,
                vector_score=0.8,
            ),
            RetrievedChunk(
                chunk_id="2104.00680:1",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Training",
                text="LoFTR training objective details.",
                fused_score=0.8,
                vector_score=0.7,
            ),
        ]

    def fake_generate(**kwargs: object) -> str:
        return "P1 [S1]\n\nP2 [S2]\n\nP3 [S1]\n\nP4 [S2][S1][S2]"

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli_module, "mlx_generate", fake_generate)

    result = runner.invoke(
        cli_module.app,
        ["eval", "--suite", str(suite_path), "--model", "mlx-community/Qwen2.5-7B-Instruct-4bit"],
    )

    assert result.exit_code == 0
    assert "PASS" in result.output


def test_eval_command_fails_on_constraint_miss(monkeypatch: object, tmp_path: Path) -> None:
    runner = CliRunner()
    settings = _settings(tmp_path)
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        (
            "- question: Explain LoFTR\n"
            "  must_include_arxiv_ids: ['2104.00680']\n"
            "  must_include_tokens: ['loftr']\n"
            "  min_sources: 1\n"
        ),
        encoding="utf-8",
    )

    class DummyQdrantStore:
        def __init__(self, url: str, collection_name: str) -> None:
            self.url = url
            self.collection_name = collection_name

    def fake_retrieve(
        self: object,
        query: str,
        top_k: int = 8,
        vector_k: int = 12,
        keyword_k: int = 12,
        require_relevance: bool = False,
        vector_score_threshold: float = 0.45,
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="2602.99999:0",
                arxiv_id="2602.99999",
                title="Unrelated",
                section_title="Intro",
                text="No matching tokens here.",
                fused_score=0.6,
                vector_score=0.6,
            )
        ]

    def fail_generate(**kwargs: object) -> str:
        raise AssertionError("LLM generation should not run when retrieval constraints fail")

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli_module, "mlx_generate", fail_generate)

    result = runner.invoke(
        cli_module.app,
        ["eval", "--suite", str(suite_path), "--model", "mlx-community/Qwen2.5-7B-Instruct-4bit"],
    )

    assert result.exit_code == 1
    assert "FAIL" in result.output
