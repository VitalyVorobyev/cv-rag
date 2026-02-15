from __future__ import annotations

import json
from pathlib import Path

from cv_rag.seeding import openalex as openalex_module
from cv_rag.seeding.openalex import extract_openalex_oa_fields


def test_extract_openalex_oa_fields_prefers_best_oa_location_pdf() -> None:
    work = {
        "best_oa_location": {
            "pdf_url": "https://example.org/best.pdf",
            "landing_page_url": "https://example.org/best",
            "license": "cc-by",
            "is_oa": True,
        },
        "primary_location": {
            "pdf_url": "https://example.org/primary.pdf",
            "landing_page_url": "https://example.org/primary",
            "license": "cc-by-nc",
            "is_oa": True,
        },
    }

    fields = extract_openalex_oa_fields(work)

    assert fields.pdf_url == "https://example.org/best.pdf"
    assert fields.landing_page_url == "https://example.org/best"
    assert fields.license == "cc-by"
    assert fields.is_oa is True
    assert fields.source == "best_oa_location"


def test_extract_openalex_oa_fields_falls_back_to_locations_oa_pdf() -> None:
    work = {
        "best_oa_location": {
            "pdf_url": None,
            "landing_page_url": "https://example.org/best",
            "license": "cc-by",
            "is_oa": True,
        },
        "primary_location": {
            "pdf_url": "",
            "landing_page_url": "https://example.org/primary",
            "license": "cc-by",
            "is_oa": True,
        },
        "locations": [
            {"is_oa": False, "pdf_url": "https://example.org/non_oa.pdf"},
            {
                "is_oa": True,
                "pdf_url": "https://example.org/from_locations.pdf",
                "landing_page_url": "https://example.org/location",
                "license": "cc0",
            },
        ],
    }

    fields = extract_openalex_oa_fields(work)

    assert fields.pdf_url == "https://example.org/from_locations.pdf"
    assert fields.landing_page_url == "https://example.org/location"
    assert fields.license == "cc0"
    assert fields.is_oa is True
    assert fields.source == "locations"


def test_extract_openalex_arxiv_id_normalizes_ids_field() -> None:
    work = {
        "ids": {
            "arxiv": "https://arxiv.org/abs/2104.00680v2",
        }
    }

    arxiv_id = openalex_module.extract_openalex_arxiv_id(work)

    assert arxiv_id == "2104.00680"


def test_resolve_dois_openalex_writes_optional_arxiv_tier_file(
    monkeypatch: object,
    tmp_path: Path,
) -> None:
    dois_path = tmp_path / "tierA_dois.txt"
    dois_path.write_text(
        "\n".join(
            [
                "10.1145/3366423.3380211",
                "10.48550/arXiv.1706.03762",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload_by_doi = {
        "10.1145/3366423.3380211": {
            "id": "https://openalex.org/W1",
            "ids": {"arxiv": "https://arxiv.org/abs/2104.00680v3"},
            "best_oa_location": {
                "pdf_url": "https://example.org/a.pdf",
                "landing_page_url": "https://example.org/a",
                "license": "cc-by",
                "is_oa": True,
            },
        },
        "10.48550/arxiv.1706.03762": {
            "id": "https://openalex.org/W2",
            "ids": {"arxiv": "arXiv:1706.03762v1"},
            "best_oa_location": {
                "pdf_url": "https://example.org/b.pdf",
                "landing_page_url": "https://example.org/b",
                "license": "cc-by",
                "is_oa": True,
            },
        },
    }

    def fake_fetch_work_json(
        *,
        client: object,
        doi: str,
        api_key: str | None,
        max_retries: int,
        backoff_start_seconds: float,
        backoff_cap_seconds: float,
        cache_dir: Path,
    ) -> tuple[dict[str, object] | None, bool]:
        _ = (
            client,
            api_key,
            max_retries,
            backoff_start_seconds,
            backoff_cap_seconds,
            cache_dir,
        )
        return payload_by_doi.get(doi), False

    monkeypatch.setattr(openalex_module, "_fetch_work_json", fake_fetch_work_json)

    out_dir = tmp_path / "out"
    tier_a_urls = tmp_path / "tierA_urls_openalex.txt"
    tier_a_arxiv = tmp_path / "tierA_arxiv_openalex.txt"

    stats = openalex_module.resolve_dois_openalex(
        dois_path=dois_path,
        out_dir=out_dir,
        user_agent="cv-rag/test",
        delay_seconds=0.0,
        tier_a_urls_path=tier_a_urls,
        tier_a_arxiv_path=tier_a_arxiv,
    )

    assert stats.resolved_records == 2
    assert stats.resolved_pdf_urls == 2
    assert stats.resolved_arxiv_ids == 2
    assert stats.tier_a_arxiv_path == tier_a_arxiv

    assert tier_a_arxiv.read_text(encoding="utf-8").splitlines() == [
        "1706.03762",
        "2104.00680",
    ]

    records = [
        json.loads(line)
        for line in (out_dir / "openalex_resolved.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all("arxiv_id" in record for record in records)
