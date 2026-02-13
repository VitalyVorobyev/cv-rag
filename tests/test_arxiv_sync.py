from cv_rag.arxiv_sync import fetch_papers_by_ids, normalize_arxiv_id


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

    monkeypatch.setattr("cv_rag.arxiv_sync._fetch_arxiv_id_feed", fail_fetch)

    papers = fetch_papers_by_ids(
        ids=["2104.00680", "1911.11763v2"],
        arxiv_api_url="https://export.arxiv.org/api/query",
        timeout_seconds=5.0,
        user_agent="test-agent",
    )

    assert [paper.arxiv_id_with_version for paper in papers] == ["2104.00680", "1911.11763v2"]
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2104.00680.pdf"
    assert papers[1].pdf_url == "https://arxiv.org/pdf/1911.11763v2.pdf"
