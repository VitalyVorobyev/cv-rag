from __future__ import annotations

from cv_rag.openalex_resolve import extract_openalex_oa_fields


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
