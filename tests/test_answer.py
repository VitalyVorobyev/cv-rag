from __future__ import annotations

from pathlib import Path
import subprocess

from typer.testing import CliRunner

import cv_rag.cli as cli_module
from cv_rag.config import Settings
from cv_rag.retrieve import RetrievedChunk, build_strict_answer_prompt


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
    assert "Every sentence must include one or more inline citations" in prompt
    assert "Every paragraph must include at least one inline citation [S#]." in prompt
    assert "The first paragraph must end with at least one citation." in prompt
    assert "Never cite a source outside S1..Sk. Never cite unrelated sources." in prompt
    assert "Do not add a separate 'Citations' footer." in prompt


def test_validate_answer_citations_rejects_uncited_paragraph() -> None:
    answer = (
        "LoFTR uses dense matching with coarse-to-fine refinement.\n\n"
        "SuperGlue uses graph neural network context [S1]."
    )
    valid, reason = cli_module._validate_answer_citations(answer, source_count=3)
    assert not valid
    assert "Paragraph 1 has no inline [S#] citation." == reason


def test_validate_answer_citations_rejects_out_of_range_citation() -> None:
    answer = "This claim cites a non-existent source [S9]."
    valid, reason = cli_module._validate_answer_citations(answer, source_count=3)
    assert not valid
    assert "outside S1..S3" in reason


def test_retrieve_for_answer_does_not_backfill_after_entity_filter() -> None:
    class DummyRetriever:
        def retrieve(
            self,
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
                    text="LoFTR coarse-to-fine pipeline.",
                    fused_score=0.9,
                ),
                RetrievedChunk(
                    chunk_id="2104.00680:1",
                    arxiv_id="2104.00680",
                    title="LoFTR",
                    section_title="Training",
                    text="LoFTR supervision and loss details.",
                    fused_score=0.8,
                ),
                RetrievedChunk(
                    chunk_id="2602.00001:0",
                    arxiv_id="2602.00001",
                    title="Unrelated diffusion paper",
                    section_title="Method",
                    text="Nothing about requested entities.",
                    fused_score=0.7,
                ),
            ]

    chunks, _ = cli_module._retrieve_for_answer(
        retriever=DummyRetriever(),  # type: ignore[arg-type]
        question="Explain LoFTR",
        k=8,
        max_per_doc=4,
        section_boost=0.0,
        entity_tokens=["loftr"],
    )

    assert len(chunks) == 2
    assert all("loftr" in f"{chunk.title} {chunk.text}".lower() for chunk in chunks)


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
        max_per_doc: int = 4,
        section_boost: float = 0.0,
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
    assert "Not found in indexed corpus. Try: cv-rag ingest-ids 2104.00680 1911.11763" in result.output


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
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        return [
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


def test_answer_refuses_comparison_when_top_doc_coverage_insufficient(
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
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="2104.00680:0",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Method",
                text="Details",
                fused_score=0.9,
                vector_score=0.7,
            ),
            RetrievedChunk(
                chunk_id="2104.00680:1",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Results",
                text="More details",
                fused_score=0.8,
                vector_score=0.6,
            ),
            RetrievedChunk(
                chunk_id="2104.00680:2",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Ablation",
                text="Even more details",
                fused_score=0.7,
                vector_score=0.5,
            ),
        ]

    def fail_run(
        cmd: list[str], check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        raise AssertionError("LLM should not be called for insufficient comparison coverage")

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli_module.subprocess, "run", fail_run)

    result = runner.invoke(
        cli_module.app,
        [
            "answer",
            "Compare LoFTR and SuperGlue",
            "--k",
            "8",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ],
    )

    assert result.exit_code == 1
    assert "Refusing to answer comparison" in result.output


def test_answer_reprompts_when_citations_missing(monkeypatch: object, tmp_path: Path) -> None:
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
        max_per_doc: int = 4,
        section_boost: float = 0.0,
    ) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="2104.00680:0",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Method",
                text="Details",
                fused_score=0.9,
                vector_score=0.8,
            ),
            RetrievedChunk(
                chunk_id="2104.00680:1",
                arxiv_id="2104.00680",
                title="LoFTR",
                section_title="Training",
                text="Training details",
                fused_score=0.85,
                vector_score=0.75,
            ),
            RetrievedChunk(
                chunk_id="1911.11763:0",
                arxiv_id="1911.11763",
                title="SuperGlue",
                section_title="Method",
                text="Details",
                fused_score=0.8,
                vector_score=0.7,
            ),
            RetrievedChunk(
                chunk_id="1911.11763:1",
                arxiv_id="1911.11763",
                title="SuperGlue",
                section_title="Training",
                text="Training details",
                fused_score=0.75,
                vector_score=0.65,
            ),
        ]

    calls: list[list[str]] = []

    def fake_run(
        cmd: list[str], check: bool, capture_output: bool, text: bool
    ) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if len(calls) == 1:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Draft answer no citations", stderr="")
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout="Revised answer [S1][S2][S3][S4][S1][S2]",
            stderr="",
        )

    monkeypatch.setattr(cli_module, "get_settings", lambda: settings)
    monkeypatch.setattr(cli_module, "QdrantStore", DummyQdrantStore)
    monkeypatch.setattr(cli_module.HybridRetriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(cli_module.subprocess, "run", fake_run)

    result = runner.invoke(
        cli_module.app,
        [
            "answer",
            "Explain LoFTR and SuperGlue",
            "--k",
            "8",
            "--model",
            "mlx-community/Qwen2.5-7B-Instruct-4bit",
        ],
    )

    assert result.exit_code == 0
    assert "Draft failed citation check; attempting repair" in result.output
    assert "Revised answer [S1][S2][S3][S4][S1][S2]" in result.output
    assert len(calls) == 2
    prompt_index = calls[1].index("--prompt") + 1
    assert "Draft to rewrite:" in calls[1][prompt_index]
    assert "Draft answer no citations" in calls[1][prompt_index]
    assert (
        "Rewrite the draft. Add inline [S#] citations to every paragraph and every non-trivial claim. "
        "Remove any claim not supported by sources."
    ) in calls[1][prompt_index]
