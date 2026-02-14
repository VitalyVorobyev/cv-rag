from cv_rag.ingest.arxiv_client import (
    PaperMetadata,
    _fetch_arxiv_id_feed,
    fetch_cs_cv_papers,
    fetch_papers_by_ids,
    normalize_arxiv_id,
)


def _paper(arxiv_id_with_version: str, title: str = "Paper") -> PaperMetadata:
    base_id = normalize_arxiv_id(arxiv_id_with_version)
    return PaperMetadata(
        arxiv_id=base_id,
        arxiv_id_with_version=arxiv_id_with_version,
        version=None,
        title=title,
        summary="summary",
        published=None,
        updated=None,
        authors=["A. Author"],
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id_with_version}.pdf",
        abs_url=f"https://arxiv.org/abs/{arxiv_id_with_version}",
    )


def test_normalize_arxiv_id_modern_abs_url() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"


def test_normalize_arxiv_id_prefixed() -> None:
    assert normalize_arxiv_id("arXiv:2401.12345v1") == "2401.12345"


def test_normalize_arxiv_id_legacy_pdf_url() -> None:
    assert normalize_arxiv_id("https://arxiv.org/pdf/cs/9901001v2.pdf") == "cs/9901001"


def test_fetch_papers_by_ids_falls_back_to_direct_urls(monkeypatch: object) -> None:
    def fail_fetch(
        arxiv_api_url: str,
        ids: list[str],
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        raise RuntimeError("simulated error")

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_id_feed", fail_fetch)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763v2"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
    )

    assert [paper.arxiv_id_with_version for paper in papers] == ["2104.00680", "1911.11763v2"]
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2104.00680.pdf"
    assert papers[1].pdf_url == "https://arxiv.org/pdf/1911.11763v2.pdf"


def test_fetch_papers_by_ids_resolves_unversioned_to_latest_when_enabled(monkeypatch: object) -> None:
    def fake_fetch(
        arxiv_api_url: str,
        ids: list[str],
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        return ",".join(ids)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        out: list[PaperMetadata] = []
        for item in (token for token in feed_text.split(",") if token):
            if "v" in item:
                out.append(_paper(arxiv_id_with_version=item, title=f"title:{item}"))
            else:
                out.append(_paper(arxiv_id_with_version=f"{item}v3", title=f"title:{item}"))
        return out

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_id_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763v2"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        resolve_unversioned_to_latest=True,
    )

    assert [paper.arxiv_id_with_version for paper in papers] == ["2104.00680v3", "1911.11763v2"]
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2104.00680v3.pdf"


def test_fetch_papers_by_ids_keeps_unversioned_when_resolution_disabled(monkeypatch: object) -> None:
    def fake_fetch(
        arxiv_api_url: str,
        ids: list[str],
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        return ",".join(ids)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        out: list[PaperMetadata] = []
        for item in (token for token in feed_text.split(",") if token):
            if "v" in item:
                out.append(_paper(arxiv_id_with_version=item, title=f"title:{item}"))
            else:
                out.append(_paper(arxiv_id_with_version=f"{item}v3", title=f"title:{item}"))
        return out

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_id_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763v2"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        resolve_unversioned_to_latest=False,
    )

    assert [paper.arxiv_id_with_version for paper in papers] == ["2104.00680", "1911.11763v2"]
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2104.00680.pdf"


def test_fetch_papers_by_ids_batches_requests(monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_fetch(
        arxiv_api_url: str,
        ids: list[str],
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        calls.append(ids)
        return ",".join(ids)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        ids = [item for item in feed_text.split(",") if item]
        return [_paper(arxiv_id_with_version=item, title=f"title:{item}") for item in ids]

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_id_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763", "2201.00001", "2201.00002", "2201.00003"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        id_batch_size=2,
    )

    assert calls == [
        ["2104.00680", "1911.11763"],
        ["2201.00001", "2201.00002"],
        ["2201.00003"],
    ]
    assert [paper.arxiv_id_with_version for paper in papers] == [
        "2104.00680",
        "1911.11763",
        "2201.00001",
        "2201.00002",
        "2201.00003",
    ]


def test_fetch_papers_by_ids_keeps_partial_metadata_on_batch_failure(monkeypatch: object) -> None:
    calls: list[list[str]] = []

    def fake_fetch(
        arxiv_api_url: str,
        ids: list[str],
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        calls.append(ids)
        if "2301.00001" in ids:
            raise RuntimeError("simulated batch failure")
        return ",".join(ids)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        ids = [item for item in feed_text.split(",") if item]
        return [_paper(arxiv_id_with_version=item, title=f"title:{item}") for item in ids]

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_id_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763", "2301.00001"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        id_batch_size=2,
    )

    assert calls == [["2104.00680", "1911.11763"], ["2301.00001"]]
    assert [paper.title for paper in papers] == [
        "title:2104.00680",
        "title:1911.11763",
        "arXiv:2301.00001",
    ]


def test_fetch_arxiv_id_feed_requests_full_batch_size(monkeypatch: object) -> None:
    captured_params: dict[str, object] = {}

    class DummyResponse:
        text = "<feed></feed>"

    def fake_retry(
        client: object,
        method: str,
        url: str,
        **request_kwargs: object,
    ) -> DummyResponse:
        params = request_kwargs.get("params")
        assert isinstance(params, dict)
        captured_params.update(params)
        return DummyResponse()

    monkeypatch.setattr("cv_rag.ingest.arxiv_client.http_request_with_retry", fake_retry)

    feed_text = _fetch_arxiv_id_feed(
        arxiv_api_url="https://export.arxiv.org/api/query",
        ids=["2104.00680", "1911.11763", "2201.00001"],
        timeout_seconds=5.0,
        user_agent="test-agent",
        max_retries=1,
        backoff_start_seconds=1.0,
        backoff_cap_seconds=2.0,
    )

    assert feed_text == "<feed></feed>"
    assert captured_params["start"] == 0
    assert captured_params["max_results"] == 3
    assert captured_params["id_list"] == "2104.00680,1911.11763,2201.00001"


def test_fetch_cs_cv_papers_skips_exact_versions(monkeypatch: object) -> None:
    calls: list[tuple[int, int]] = []

    def fake_fetch(
        start: int,
        max_results: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        calls.append((start, max_results))
        return str(start)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        if feed_text == "0":
            return [
                _paper("2104.00680v1"),
                _paper("2104.00680v2"),
                _paper("1911.11763v1"),
            ]
        return []

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_api_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_cs_cv_papers(
        limit=3,
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        skip_arxiv_id_with_version={"2104.00680v1"},
        max_scan_results=6,
    )

    assert [paper.arxiv_id_with_version for paper in papers] == ["2104.00680v2", "1911.11763v1"]
    assert calls[0][0] == 0


def test_fetch_cs_cv_papers_fills_limit_with_pagination(monkeypatch: object) -> None:
    calls: list[tuple[int, int]] = []

    def fake_fetch(
        start: int,
        max_results: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        calls.append((start, max_results))
        return str(start)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        if feed_text == "0":
            return [
                _paper("2000.00001v1"),
                _paper("2000.00002v1"),
                _paper("2000.00003v1"),
            ]
        if feed_text == "3":
            return [
                _paper("2000.00004v1"),
                _paper("2000.00005v1"),
            ]
        return []

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_api_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_cs_cv_papers(
        limit=3,
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        skip_arxiv_id_with_version={"2000.00001v1", "2000.00002v1"},
    )

    assert [paper.arxiv_id_with_version for paper in papers] == [
        "2000.00003v1",
        "2000.00004v1",
        "2000.00005v1",
    ]
    assert [start for start, _ in calls] == [0, 3]


def test_fetch_cs_cv_papers_respects_scan_cap(monkeypatch: object) -> None:
    calls: list[tuple[int, int]] = []

    def fake_fetch(
        start: int,
        max_results: int,
        arxiv_api_url: str,
        timeout_seconds: float,
        user_agent: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str:
        calls.append((start, max_results))
        return str(start)

    def fake_parse(feed_text: str) -> list[PaperMetadata]:
        if feed_text == "0":
            return [_paper("3000.00001v1"), _paper("3000.00002v1")]
        if feed_text == "2":
            return [_paper("3000.00003v1"), _paper("3000.00004v1")]
        return [_paper("3000.00005v1")]

    monkeypatch.setattr("cv_rag.ingest.arxiv_client._fetch_arxiv_api_feed", fake_fetch)
    monkeypatch.setattr("cv_rag.ingest.arxiv_client._parse_api_feed", fake_parse)

    papers = fetch_cs_cv_papers(
        limit=5,
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
        skip_arxiv_id_with_version={"3000.00001v1", "3000.00002v1", "3000.00003v1", "3000.00004v1"},
        max_scan_results=4,
    )

    assert papers == []
    assert [start for start, _ in calls] == [0]
