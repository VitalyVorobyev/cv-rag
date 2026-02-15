from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from cv_rag.answer.models import AnswerEvent
from cv_rag.interfaces.api.deps import get_app_settings, get_retriever, get_sqlite_store
from cv_rag.interfaces.api.routers import answer, health, papers, search, stats
from cv_rag.retrieval.models import RetrievedChunk

# ===========================================================================
# Fixtures
# ===========================================================================


def _make_settings(tmp_path: Path) -> MagicMock:
    s = MagicMock()
    s.qdrant_url = "http://localhost:6333"
    s.grobid_url = "http://localhost:8070"
    s.ollama_url = "http://localhost:11434"
    s.qdrant_collection = "cv_papers"
    s.pdf_dir = tmp_path / "pdfs"
    s.tei_dir = tmp_path / "tei"
    s.pdf_dir.mkdir()
    s.tei_dir.mkdir()
    return s


def _make_app(
    routers: list,
    *,
    mock_retriever: MagicMock | None = None,
    mock_sqlite: MagicMock | None = None,
    mock_settings: MagicMock | None = None,
) -> FastAPI:
    app = FastAPI()
    for r in routers:
        app.include_router(r, prefix="/api")

    if mock_retriever is not None:
        app.dependency_overrides[get_retriever] = lambda: mock_retriever
    if mock_sqlite is not None:
        app.dependency_overrides[get_sqlite_store] = lambda: mock_sqlite
    if mock_settings is not None:
        app.dependency_overrides[get_app_settings] = lambda: mock_settings

    return app


# ===========================================================================
# Health router
# ===========================================================================


def test_health_all_ok(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    app = _make_app([health.router], mock_settings=settings)

    qdrant_resp = MagicMock()
    qdrant_resp.json.return_value = {"version": "1.8.0"}
    qdrant_resp.raise_for_status = MagicMock()

    grobid_alive = MagicMock()
    grobid_alive.raise_for_status = MagicMock()
    grobid_alive.text = "true"
    grobid_version = MagicMock()
    grobid_version.status_code = 200
    grobid_version.text = "0.8.0"

    ollama_version = MagicMock()
    ollama_version.raise_for_status = MagicMock()
    ollama_version.json.return_value = {"version": "0.1.0"}
    ollama_tags = MagicMock()
    ollama_tags.raise_for_status = MagicMock()
    ollama_tags.json.return_value = {"models": [{"name": "m1"}, {"name": "m2"}]}

    def fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
        if "6333" in url:
            return qdrant_resp
        if "isalive" in url:
            return grobid_alive
        if "8070" in url and "version" in url:
            return grobid_version
        if "11434" in url and "version" in url:
            return ollama_version
        if "tags" in url:
            return ollama_tags
        return MagicMock()

    with (
        patch("cv_rag.interfaces.api.routers.health.httpx.get", side_effect=fake_get),
        patch("cv_rag.interfaces.api.routers.health.QdrantStore") as mock_qs,
    ):
        mock_qs.return_value.client.get_collections.return_value = MagicMock()
        client = TestClient(app)
        resp = client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["services"]) == 3
    statuses = {s["service"]: s["status"] for s in data["services"]}
    assert statuses["Qdrant"] == "ok"
    assert statuses["GROBID"] == "ok"
    assert statuses["Ollama"] == "ok"


def test_health_service_failure(tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)
    app = _make_app([health.router], mock_settings=settings)

    with (
        patch("cv_rag.interfaces.api.routers.health.httpx.get", side_effect=Exception("timeout")),
        patch("cv_rag.interfaces.api.routers.health.QdrantStore", side_effect=Exception("timeout")),
    ):
        client = TestClient(app)
        resp = client.get("/api/health")

    assert resp.status_code == 200
    data = resp.json()
    statuses = {s["service"]: s["status"] for s in data["services"]}
    assert statuses["Qdrant"] == "fail"
    assert statuses["GROBID"] == "fail"
    assert statuses["Ollama"] == "fail"


# ===========================================================================
# Search router
# ===========================================================================


def test_search_returns_chunks() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        RetrievedChunk(
            chunk_id="2104.00680:0",
            arxiv_id="2104.00680",
            title="LoFTR",
            section_title="Abstract",
            text="Feature matching.",
            fused_score=0.85,
            vector_score=0.9,
            keyword_score=0.7,
            sources=["vector", "keyword"],
        ),
    ]

    app = _make_app([search.router], mock_retriever=retriever)
    client = TestClient(app)
    resp = client.post("/api/search", json={"query": "feature matching", "top_k": 5})

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["chunks"]) == 1
    assert data["chunks"][0]["chunk_id"] == "2104.00680:0"
    assert data["query"] == "feature matching"
    assert "elapsed_ms" in data

    retriever.retrieve.assert_called_once()
    call_kwargs = retriever.retrieve.call_args[1]
    assert call_kwargs["query"] == "feature matching"
    assert call_kwargs["top_k"] == 5


def test_search_empty_results() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = []

    app = _make_app([search.router], mock_retriever=retriever)
    client = TestClient(app)
    resp = client.post("/api/search", json={"query": "nonexistent"})

    assert resp.status_code == 200
    assert resp.json()["chunks"] == []


# ===========================================================================
# Papers router
# ===========================================================================


def _make_sqlite_with_papers() -> MagicMock:
    """Build a mock SQLiteStore whose conn.execute returns paper rows."""
    store = MagicMock()
    conn = MagicMock()
    store.conn = conn

    paper_data = {
        "arxiv_id": "2104.00680",
        "title": "LoFTR",
        "summary": "Feature matching",
        "published": "2021-04-01",
        "updated": "2021-04-01",
        "authors_json": json.dumps(["Author A", "Author B"]),
        "pdf_url": "https://arxiv.org/pdf/2104.00680v1.pdf",
        "abs_url": "https://arxiv.org/abs/2104.00680v1",
        "chunk_count": 5,
        "tier": 0,
        "citation_count": 321,
        "venue": "CVPR",
    }

    def _make_row(data: dict) -> MagicMock:
        row = MagicMock()
        row.__getitem__ = lambda self, key, d=data: d[key]
        return row

    count_row = MagicMock()
    count_row.__getitem__ = lambda self, idx: 1

    def fake_execute(query, params=()):  # type: ignore[no-untyped-def]
        result = MagicMock()
        q = query.strip()
        if q.startswith("SELECT COUNT"):
            result.fetchone.return_value = count_row
        elif "FROM chunks" in query and "WHERE arxiv_id" in query:
            result.fetchall.return_value = []
        elif "WHERE p.arxiv_id = ?" in query:
            result.fetchone.return_value = _make_row(paper_data)
        else:
            result.fetchall.return_value = [_make_row(paper_data)]
        return result

    conn.execute = fake_execute
    return store


def test_list_papers_returns_papers() -> None:
    store = _make_sqlite_with_papers()
    app = _make_app([papers.router], mock_sqlite=store)
    client = TestClient(app)
    resp = client.get("/api/papers")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["papers"]) == 1
    assert data["papers"][0]["arxiv_id"] == "2104.00680"


def test_get_paper_not_found() -> None:
    store = MagicMock()
    conn = MagicMock()
    store.conn = conn

    def fake_execute(query, params=()):  # type: ignore[no-untyped-def]
        result = MagicMock()
        result.fetchone.return_value = None
        result.fetchall.return_value = []
        return result

    conn.execute = fake_execute

    app = _make_app([papers.router], mock_sqlite=store)
    client = TestClient(app)
    resp = client.get("/api/papers/9999.99999")

    assert resp.status_code == 404


# ===========================================================================
# Stats router
# ===========================================================================


def test_stats_returns_counts(tmp_path: Path) -> None:
    store = MagicMock()
    conn = MagicMock()
    store.conn = conn
    settings = _make_settings(tmp_path)

    # Create some dummy files
    (tmp_path / "pdfs" / "a.pdf").write_bytes(b"")
    (tmp_path / "tei" / "a.tei.xml").write_bytes(b"")

    def fake_execute(query, params=()):  # type: ignore[no-untyped-def]
        result = MagicMock()
        q = query.strip()

        if "GROUP BY tier" in q:
            tier_row = MagicMock()
            tier_row.__getitem__ = lambda self, key: {"tier": 0, "count": 3}[key]
            result.fetchall.return_value = [tier_row]
        elif "GROUP BY venue" in q:
            venue_row = MagicMock()
            venue_row.__getitem__ = lambda self, key: {"venue": "CVPR", "count": 5}[key]
            result.fetchall.return_value = [venue_row]
        else:
            count_row = MagicMock()
            count_row.__getitem__ = lambda self, idx: 10
            result.fetchone.return_value = count_row
        return result

    conn.execute = fake_execute

    app = _make_app([stats.router], mock_sqlite=store, mock_settings=settings)
    client = TestClient(app)
    resp = client.get("/api/stats")

    assert resp.status_code == 200
    data = resp.json()
    assert data["papers_count"] == 10
    assert data["pdf_files"] == 1
    assert data["tei_files"] == 1
    assert "tier_distribution" in data


# ===========================================================================
# Papers router — _parse_authors
# ===========================================================================


def test_parse_authors_json_array() -> None:
    from cv_rag.interfaces.api.routers.papers import _parse_authors

    assert _parse_authors(json.dumps(["Alice", "Bob"])) == ["Alice", "Bob"]


def test_parse_authors_comma_fallback() -> None:
    from cv_rag.interfaces.api.routers.papers import _parse_authors

    assert _parse_authors("Alice, Bob") == ["Alice", "Bob"]


def test_parse_authors_none() -> None:
    from cv_rag.interfaces.api.routers.papers import _parse_authors

    assert _parse_authors(None) == []


def test_parse_authors_empty_string() -> None:
    from cv_rag.interfaces.api.routers.papers import _parse_authors

    assert _parse_authors("") == []


# ===========================================================================
# Answer router
# ===========================================================================


def test_answer_router_done_payload_includes_v2_route_fields(tmp_path: Path) -> None:
    retriever = MagicMock()
    settings = _make_settings(tmp_path)
    app = _make_app([answer.router], mock_retriever=retriever, mock_settings=settings)

    class FakeAnswerService:
        def __init__(self, retriever: object, settings: object) -> None:
            _ = (retriever, settings)

        def stream(self, request: object):  # type: ignore[no-untyped-def]
            _ = request
            yield AnswerEvent(
                event="done",
                data={
                    "answer": "P1: Test [S1]\\n\\nP2: Test [S1]\\n\\nP3: Test [S1]\\n\\nP4: Test [S1]",
                    "sources": [],
                    "route": {
                        "mode": "explain",
                        "targets": [],
                        "k": 8,
                        "max_per_doc": 4,
                        "confidence": 0.9,
                        "notes": "test route",
                        "preface": None,
                        "reason_codes": ["default_explain"],
                        "policy_version": "v2",
                    },
                    "citation_valid": True,
                    "citation_reason": "",
                    "elapsed_ms": 2.3,
                },
            )

    with patch("cv_rag.interfaces.api.routers.answer.AnswerService", FakeAnswerService):
        client = TestClient(app)
        resp = client.post(
            "/api/answer",
            json={
                "question": "Explain LoFTR",
                "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                "mode": "auto",
            },
        )

    assert resp.status_code == 200
    body = resp.text
    assert '\"policy_version\": \"v2\"' in body
    assert '\"reason_codes\": [\"default_explain\"]' in body
