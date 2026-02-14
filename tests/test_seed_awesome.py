from __future__ import annotations

import json
from pathlib import Path

from cv_rag.seed_awesome import (
    AwesomeDoiSeedRecord,
    AwesomeSeedRecord,
    extract_arxiv_matches,
    extract_doi_matches,
    write_seed_outputs,
)


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


def test_extract_doi_matches_handles_urls_prefix_and_trailing_punctuation() -> None:
    text = (
        "Paper: https://doi.org/10.1145/3366423.3380211 and "
        "DOI:10.1145/3366423.3380211). "
        "The same URL has bare DOI overlap 10.1145/3366423.3380211 inside."
    )

    matches = extract_doi_matches(text)

    assert [match.doi for match in matches] == [
        "10.1145/3366423.3380211",
        "10.1145/3366423.3380211",
        "10.1145/3366423.3380211",
    ]
    assert matches[0].raw_match == "https://doi.org/10.1145/3366423.3380211"
    assert matches[1].raw_match == "DOI:10.1145/3366423.3380211"


def test_write_seed_outputs_writes_sorted_unique_tier_files(tmp_path: Path) -> None:
    text = (
        "Refs: arXiv:2104.00680v2, https://arxiv.org/pdf/1911.11763.pdf, "
        "doi:10.1145/3366423.3380211 and https://doi.org/10.48550/ARXIV.1706.03762."
    )
    arxiv_matches = extract_arxiv_matches(text)
    doi_matches = extract_doi_matches(text)

    arxiv_records = [
        AwesomeSeedRecord(
            base_id=match.base_id,
            arxiv_id=match.arxiv_id,
            source_repo="owner/repo",
            source_url="https://github.com/owner/repo",
            found_in="README.md",
            raw_match=match.raw_match,
        )
        for match in arxiv_matches
    ]
    # Add duplicate base-id record to verify unique sorting in tier files.
    arxiv_records.append(
        AwesomeSeedRecord(
            base_id="2104.00680",
            arxiv_id="2104.00680",
            source_repo="owner/repo2",
            source_url="https://github.com/owner/repo2",
            found_in="README.md",
            raw_match="arXiv:2104.00680",
        )
    )

    doi_records = [
        AwesomeDoiSeedRecord(
            doi=match.doi,
            source_repo="owner/repo",
            source_url="https://github.com/owner/repo",
            found_in="README.md",
            raw_match=match.raw_match,
        )
        for match in doi_matches
    ]
    doi_records.append(
        AwesomeDoiSeedRecord(
            doi="10.1145/3366423.3380211",
            source_repo="owner/repo2",
            source_url="https://github.com/owner/repo2",
            found_in="README.md",
            raw_match="doi:10.1145/3366423.3380211",
        )
    )

    out_dir = tmp_path / "out"
    tier_seed = tmp_path / "tierA_seed.txt"
    tier_arxiv = tmp_path / "tierA_arxiv.txt"
    tier_dois = tmp_path / "tierA_dois.txt"

    jsonl_path, doi_jsonl_path, _, _, unique_arxiv_count, unique_doi_count = write_seed_outputs(
        arxiv_records=arxiv_records,
        doi_records=doi_records,
        out_dir=out_dir,
        tier_a_seed_path=tier_seed,
        tier_a_arxiv_path=tier_arxiv,
        tier_a_dois_path=tier_dois,
    )

    assert jsonl_path.exists()
    assert doi_jsonl_path.exists()
    assert unique_arxiv_count == 2
    assert unique_doi_count == 2

    assert tier_arxiv.read_text(encoding="utf-8").splitlines() == ["1911.11763", "2104.00680"]
    assert tier_seed.read_text(encoding="utf-8").splitlines() == ["1911.11763", "2104.00680"]
    assert tier_dois.read_text(encoding="utf-8").splitlines() == [
        "10.1145/3366423.3380211",
        "10.48550/arxiv.1706.03762",
    ]

    first_record = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert "base_id" in first_record
    first_doi_record = json.loads(doi_jsonl_path.read_text(encoding="utf-8").splitlines()[0])
    assert "doi" in first_doi_record
