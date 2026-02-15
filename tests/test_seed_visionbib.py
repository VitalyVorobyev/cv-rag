from __future__ import annotations

import json
from pathlib import Path

from cv_rag.seeding import visionbib as visionbib_module


def test_load_visionbib_sources_parses_three_token_ranges(tmp_path: Path) -> None:
    sources_path = tmp_path / "visionbib_sources.txt"
    sources_path.write_text(
        "\n".join(
            [
                "https://www.visionbib.com/bibliography/",
                "compute 42 44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = visionbib_module.load_visionbib_sources(sources_path)

    assert spec.prefix_url == "https://www.visionbib.com/bibliography/"
    assert len(spec.ranges) == 1
    assert spec.ranges[0].stem == "compute"
    assert spec.ranges[0].start == 42
    assert spec.ranges[0].end == 44


def test_load_visionbib_sources_parses_compact_range_tokens(tmp_path: Path) -> None:
    sources_path = tmp_path / "visionbib_sources.txt"
    sources_path.write_text(
        "\n".join(
            [
                "https://www.visionbib.com/bibliography/",
                "compute42 compute44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    spec = visionbib_module.load_visionbib_sources(sources_path)
    assert len(spec.ranges) == 1
    assert spec.ranges[0].stem == "compute"
    assert spec.ranges[0].start == 42
    assert spec.ranges[0].end == 44


def test_load_visionbib_sources_rejects_mixed_compact_stems(tmp_path: Path) -> None:
    sources_path = tmp_path / "visionbib_sources.txt"
    sources_path.write_text(
        "\n".join(
            [
                "https://www.visionbib.com/bibliography/",
                "compute42 twod44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        visionbib_module.load_visionbib_sources(sources_path)
    except ValueError as exc:
        assert "Range stems do not match" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mixed compact range stems")


def test_expand_page_urls_is_inclusive() -> None:
    spec = visionbib_module.VisionBibSourceSpec(
        prefix_url="https://www.visionbib.com/bibliography/",
        ranges=[visionbib_module.VisionBibPageRange(stem="compute", start=42, end=44)],
    )

    pages = visionbib_module.expand_page_urls(spec)
    assert [page.page_url for page in pages] == [
        "https://www.visionbib.com/bibliography/compute42.html",
        "https://www.visionbib.com/bibliography/compute43.html",
        "https://www.visionbib.com/bibliography/compute44.html",
    ]


def test_seed_visionbib_sources_extracts_links_and_writes_outputs(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    sources_path = tmp_path / "visionbib_sources.txt"
    sources_path.write_text(
        "\n".join(
            [
                "https://www.visionbib.com/bibliography/",
                "compute 42 42",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    page_html = """
    <html><body>
      <a href="https://doi.org/10.1145/3366423.3380211">paper doi</a>
      <a href="paper.pdf">local pdf</a>
      <a href="https://arxiv.org/abs/2104.00680v2">arxiv link</a>
      <a href="https://doi.org/10.1145/3366423.3380211">duplicate doi link</a>
      Also discussed in DOI:10.1145/3366423.3380211.
    </body></html>
    """

    def fake_request_optional_page(
        *,
        client: object,
        url: str,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
    ) -> str | None:
        _ = (client, url, max_retries, backoff_start_seconds, backoff_cap_seconds)
        return page_html

    monkeypatch.setattr(visionbib_module, "_request_optional_page", fake_request_optional_page)

    out_dir = tmp_path / "out"
    tier_dois = tmp_path / "tierA_dois_visionbib.txt"
    tier_urls = tmp_path / "tierA_urls_visionbib.txt"
    tier_arxiv = tmp_path / "tierA_arxiv_visionbib.txt"

    stats = visionbib_module.seed_visionbib_sources(
        sources_path=sources_path,
        out_dir=out_dir,
        user_agent="cv-rag/test",
        delay_seconds=0.0,
        tier_a_dois_path=tier_dois,
        tier_a_urls_path=tier_urls,
        tier_a_arxiv_path=tier_arxiv,
    )

    assert stats.pages_requested == 1
    assert stats.pages_succeeded == 1
    assert stats.pages_failed == 0
    assert stats.unique_dois == 1
    assert stats.unique_pdf_urls == 1
    assert stats.unique_arxiv_ids == 1

    assert tier_dois.read_text(encoding="utf-8").splitlines() == ["10.1145/3366423.3380211"]
    assert tier_urls.read_text(encoding="utf-8").splitlines() == [
        "https://www.visionbib.com/bibliography/paper.pdf"
    ]
    assert tier_arxiv.read_text(encoding="utf-8").splitlines() == ["2104.00680"]

    doi_records = [
        json.loads(line)
        for line in (out_dir / "visionbib_seed_doi.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    url_records = [
        json.loads(line)
        for line in (out_dir / "visionbib_seed_url.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    arxiv_records = [
        json.loads(line)
        for line in (out_dir / "visionbib_seed_arxiv.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert len(doi_records) == 1
    assert doi_records[0]["kind"] == "doi"
    assert len(url_records) == 1
    assert url_records[0]["kind"] == "pdf"
    assert len(arxiv_records) == 1
    assert arxiv_records[0]["kind"] == "arxiv"
