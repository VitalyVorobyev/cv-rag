from __future__ import annotations

from cv_rag.seed_awesome import extract_arxiv_matches


def test_extract_arxiv_matches_from_abs_and_pdf_urls() -> None:
    text = (
        "See https://arxiv.org/abs/2104.00680v2 for details, "
        "plus https://arxiv.org/pdf/1911.11763.pdf and "
        "https://arxiv.org/pdf/2301.00001v3.pdf."
    )

    matches = extract_arxiv_matches(text)

    assert [(match.base_id, match.arxiv_id, match.version) for match in matches] == [
        ("2104.00680", "2104.00680v2", 2),
        ("1911.11763", "1911.11763", None),
        ("2301.00001", "2301.00001v3", 3),
    ]
    assert matches[0].raw_match == "https://arxiv.org/abs/2104.00680v2"
    assert matches[1].raw_match == "https://arxiv.org/pdf/1911.11763.pdf"


def test_extract_arxiv_matches_from_arxiv_prefix_text() -> None:
    text = "Related: arXiv:2104.00680v4; baseline in arxiv 1911.11763."

    matches = extract_arxiv_matches(text)

    assert [(match.base_id, match.arxiv_id, match.version) for match in matches] == [
        ("2104.00680", "2104.00680v4", 4),
        ("1911.11763", "1911.11763", None),
    ]
    assert matches[0].raw_match == "arXiv:2104.00680v4"
    assert matches[1].raw_match == "arxiv 1911.11763"


def test_extract_arxiv_matches_supports_legacy_ids() -> None:
    text = (
        "Classic link: https://arxiv.org/abs/cs/9901001v2 and inline "
        "arxiv cs/9901001v3."
    )

    matches = extract_arxiv_matches(text)

    assert [(match.base_id, match.arxiv_id, match.version) for match in matches] == [
        ("cs/9901001", "cs/9901001v2", 2),
        ("cs/9901001", "cs/9901001v3", 3),
    ]
