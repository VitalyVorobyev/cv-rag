from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any, cast

import pytest

from cv_rag.answer.models import AnswerRunRequest
from cv_rag.answer.routing import AnswerMode
from cv_rag.answer.service import AnswerService
from cv_rag.retrieval.models import RetrievedChunk
from cv_rag.shared.errors import CitationValidationError, GenerationError
from cv_rag.shared.settings import Settings


class _FakeRetriever:
    def __init__(self, responses: list[list[RetrievedChunk]], *, irrelevant: bool = False) -> None:
        self._responses = responses
        self._irrelevant = irrelevant
        self.retrieve_calls: list[dict[str, object]] = []

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
        self.retrieve_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "vector_k": vector_k,
                "keyword_k": keyword_k,
                "require_relevance": require_relevance,
                "vector_score_threshold": vector_score_threshold,
                "max_per_doc": max_per_doc,
                "section_boost": section_boost,
            }
        )
        call_index = len(self.retrieve_calls) - 1
        if not self._responses:
            return []
        if call_index >= len(self._responses):
            return self._responses[-1]
        return self._responses[call_index]

    def _is_irrelevant_result(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        vector_score_threshold: float,
    ) -> bool:
        _ = query
        _ = candidates
        _ = vector_score_threshold
        return self._irrelevant


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        pdf_dir=tmp_path / "data" / "pdfs",
        tei_dir=tmp_path / "data" / "tei",
        metadata_dir=tmp_path / "data" / "metadata",
        metadata_json_path=tmp_path / "data" / "metadata" / "arxiv_cs_cv.json",
        sqlite_path=tmp_path / "cv_rag.sqlite3",
    )


def _chunk(
    *,
    chunk_id: str,
    arxiv_id: str = "2104.00680",
    title: str = "LoFTR",
    section_title: str = "Method",
    text: str = "LoFTR supervision details.",
    fused_score: float = 0.9,
    vector_score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        arxiv_id=arxiv_id,
        title=title,
        section_title=section_title,
        text=text,
        fused_score=fused_score,
        vector_score=vector_score,
    )


def _request(**overrides: Any) -> AnswerRunRequest:
    data = {
        "question": "Explain LoFTR supervision details.",
        "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "mode": "auto",
        "router_strategy": "rules",
    }
    data.update(overrides)
    return AnswerRunRequest(**data)


def _valid_answer() -> str:
    return (
        "P1: LoFTR uses coarse-to-fine matching with dense correspondences [S1].\n\n"
        "P2: The supervision objective aligns local and global matching cues [S2].\n\n"
        "P3: Training emphasizes robust matching under challenging viewpoints [S1].\n\n"
        "P4: The reported design reduces hand-crafted matching heuristics [S2]."
    )


def _empty_stream_generate(**kwargs: object) -> Generator[str, None, None]:
    _ = kwargs
    if False:
        yield ""


def test_answer_service_run_happy_path_auto_rules(tmp_path: Path) -> None:
    prelim_chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    final_chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    retriever = _FakeRetriever([prelim_chunks, final_chunks])

    def fake_generate(**kwargs: object) -> str:
        _ = kwargs
        return _valid_answer()

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=_empty_stream_generate,
    )

    result = service.run(_request())

    assert result.answer == _valid_answer()
    assert result.citation_valid is True
    assert result.citation_reason == ""
    assert result.route.mode is AnswerMode.EXPLAIN
    assert len(result.sources) == 2
    assert result.warnings == []
    assert len(retriever.retrieve_calls) == 2


def test_answer_service_run_repair_loop_succeeds(tmp_path: Path) -> None:
    chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    retriever = _FakeRetriever([chunks, chunks])
    prompts: list[str] = []

    def fake_generate(**kwargs: object) -> str:
        prompts.append(str(kwargs["prompt"]))
        if len(prompts) == 1:
            return "Draft with no citations at all."
        return _valid_answer()

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=_empty_stream_generate,
    )

    result = service.run(_request())

    assert result.citation_valid is True
    assert result.answer == _valid_answer()
    assert "Draft failed citation check; attempting repair" in result.warnings
    assert len(prompts) == 2
    assert "Draft to rewrite:" in prompts[1]
    assert "Draft with no citations at all." in prompts[1]


def test_answer_service_run_raises_when_repair_fails(tmp_path: Path) -> None:
    chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    retriever = _FakeRetriever([chunks, chunks])

    first_draft = "Initial draft with no inline references."
    second_draft = "Repair draft still missing references."
    call_count = {"n": 0}

    def fake_generate(**kwargs: object) -> str:
        _ = kwargs
        call_count["n"] += 1
        if call_count["n"] == 1:
            return first_draft
        return second_draft

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=_empty_stream_generate,
    )

    with pytest.raises(CitationValidationError) as exc_info:
        service.run(_request())

    assert call_count["n"] == 2
    assert "Paragraph 1 has no inline [S#] citation." in exc_info.value.reason
    assert exc_info.value.draft == first_draft


def test_answer_service_stream_emits_route_sources_tokens_done(tmp_path: Path) -> None:
    chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    retriever = _FakeRetriever([chunks, chunks])

    token_chunks = [
        "P1: LoFTR uses coarse-to-fine matching [S1].\n\nP2: Supervision aligns cues [S2].\n\n",
        "P3: Training improves geometric robustness [S1].\n\nP4: The method removes heuristics [S2].",
    ]

    def fake_stream_generate(**kwargs: object) -> Generator[str, None, None]:
        _ = kwargs
        yield from token_chunks

    def fake_generate(**kwargs: object) -> str:
        _ = kwargs
        return _valid_answer()

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=fake_stream_generate,
    )

    events = list(service.stream(_request()))
    event_names = [event.event for event in events]

    assert event_names[0] == "route"
    assert event_names[1] == "sources"
    assert event_names[-1] == "done"
    token_events = [event for event in events if event.event == "token"]
    assert len(token_events) >= 1
    assert "".join(str(event.data) for event in token_events) == "".join(token_chunks)
    done_payload = cast(dict[str, object], events[-1].data)
    assert done_payload["citation_valid"] is True


def test_answer_service_stream_falls_back_after_generation_error(tmp_path: Path) -> None:
    chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
    ]
    retriever = _FakeRetriever([chunks, chunks])
    fallback_answer = _valid_answer()
    fallback_calls = {"n": 0}

    def fake_stream_generate(**kwargs: object) -> Generator[str, None, None]:
        _ = kwargs
        raise GenerationError("stream generation failed")
        yield ""

    def fake_generate(**kwargs: object) -> str:
        _ = kwargs
        fallback_calls["n"] += 1
        return fallback_answer

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=fake_stream_generate,
    )

    events = list(service.stream(_request()))
    token_events = [event for event in events if event.event == "token"]
    done_payload = cast(dict[str, object], events[-1].data)

    assert fallback_calls["n"] == 1
    assert len(token_events) >= 1
    assert any(str(event.data) == fallback_answer for event in token_events)
    assert events[-1].event == "done"
    assert done_payload["citation_valid"] is True


def test_answer_service_stream_emits_error_for_compare_refusal(tmp_path: Path) -> None:
    chunks = [
        _chunk(chunk_id="2104.00680:0"),
        _chunk(chunk_id="2104.00680:1", section_title="Training", fused_score=0.85),
        _chunk(chunk_id="2104.00680:2", section_title="Results", fused_score=0.8),
    ]
    retriever = _FakeRetriever([chunks, chunks])

    def fake_generate(**kwargs: object) -> str:
        _ = kwargs
        return _valid_answer()

    service = AnswerService(
        retriever=cast(Any, retriever),
        settings=_settings(tmp_path),
        generate_fn=fake_generate,
        stream_generate_fn=_empty_stream_generate,
    )

    events = list(
        service.stream(
            _request(
                question="Compare LoFTR and SuperGlue.",
                mode="compare",
            )
        )
    )

    assert len(events) == 1
    assert events[0].event == "error"
    error_payload = cast(dict[str, str], events[0].data)
    assert "Refusing to answer comparison" in error_payload["message"]
