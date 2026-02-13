from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

import cv_rag.cli as cli_module
from cv_rag.config import Settings
from cv_rag.retrieve import NoRelevantSourcesError, RetrievedChunk, build_strict_answer_prompt


def test_build_strict_answer_prompt_includes_sources_and_rules() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="a:0",
            arxiv_id="2401.00001",
            title="Paper A",
            section_title="Methods",
            text="Patch embeddings are learned with a linear projection.",
            fused_score=0.5,
        ),
        RetrievedChunk(
            chunk_id="b:0",
            arxiv_id="2401.00002",
            title="Paper B",
            section_title="Results",
            text="The model improves top-1 accuracy by 1.2 points.",
            fused_score=0.4,
        ),
    ]

    prompt = build_strict_answer_prompt("How do patch embeddings work?", chunks)

    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert "Only use information supported by the sources." in prompt
    assert "Every non-trivial claim must include citations like [S3][S7]." in prompt


def test_answer_command_errors_when_no_sources_retrieved(
    monkeypatch: object, tmp_path: Path
) -> None:
    runner = CliRunner()

    settings = Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
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
    ) -> list[RetrievedChunk]:
        return []

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)

    result = runner.invoke(
        cli_module.app,
        [
            "answer",
            "vision transformer patch embedding",
            "--k",
            "8",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ],
    )

    assert result.exit_code == 1
    assert "No sources retrieved for this question" in result.output


def test_answer_command_refuses_when_no_relevant_sources(
    monkeypatch: object, tmp_path: Path
) -> None:
    runner = CliRunner()

    settings = Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
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
    ) -> list[RetrievedChunk]:
        candidates = [
            RetrievedChunk(
                chunk_id="2602.00001:0",
                arxiv_id="2602.00001",
                title="Unrelated Paper",
                section_title="Introduction",
                text="This discusses unrelated methods.",
                fused_score=0.8,
                vector_score=0.1,
            ),
            RetrievedChunk(
                chunk_id="2602.00002:0",
                arxiv_id="2602.00002",
                title="Another Unrelated Paper",
                section_title="Method",
                text="No mention of target terms.",
                fused_score=0.7,
                vector_score=0.1,
            ),
        ]
        raise NoRelevantSourcesError(candidates)

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)

    result = runner.invoke(
        cli_module.app,
        [
            "answer",
            "Explain LoFTR vs SuperGlue",
            "--k",
            "8",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ],
    )

    assert result.exit_code == 1
    assert "Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763" in result.output
