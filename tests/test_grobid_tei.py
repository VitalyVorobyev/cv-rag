from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from cv_rag.ingest.tei_extract import _node_text, _normalize_text, extract_sections

# ===========================================================================
# TEI extract — _normalize_text
# ===========================================================================


def test_normalize_text_collapses_whitespace() -> None:
    assert _normalize_text("  hello   world  ") == "hello world"


def test_normalize_text_replaces_tabs_newlines() -> None:
    assert _normalize_text("a\tb\nc") == "a b c"


def test_normalize_text_already_normal() -> None:
    assert _normalize_text("clean text") == "clean text"


# ===========================================================================
# TEI extract — _node_text
# ===========================================================================


def test_node_text_none_returns_empty() -> None:
    assert _node_text(None) == ""


# ===========================================================================
# TEI extract — extract_sections
# ===========================================================================


def test_extract_sections_empty_string() -> None:
    assert extract_sections("") == []
    assert extract_sections("   ") == []


TEI_NS = 'xmlns="http://www.tei-c.org/ns/1.0"'


def test_extract_sections_abstract_only() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc><abstract>
            <p>First paragraph.</p>
            <p>Second paragraph.</p>
        </abstract></profileDesc></teiHeader>
        <text><body/></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 1
    assert sections[0].title == "Abstract"
    assert "First paragraph." in sections[0].text
    assert "Second paragraph." in sections[0].text


def test_extract_sections_body_named_divs() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc/></teiHeader>
        <text><body>
            <div><head>Introduction</head><p>Intro text.</p></div>
            <div><head>Method</head><p>Method text.</p></div>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 2
    assert sections[0].title == "Introduction"
    assert sections[0].text == "Intro text."
    assert sections[1].title == "Method"


def test_extract_sections_unnamed_divs_get_fallback_titles() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc/></teiHeader>
        <text><body>
            <div><p>Content A.</p></div>
            <div><p>Content B.</p></div>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 2
    assert sections[0].title == "Section 1"
    assert sections[1].title == "Section 2"


def test_extract_sections_abstract_plus_body() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc><abstract>
            <p>Abstract text.</p>
        </abstract></profileDesc></teiHeader>
        <text><body>
            <div><head>Results</head><p>Result text.</p></div>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 2
    assert sections[0].title == "Abstract"
    assert sections[1].title == "Results"


def test_extract_sections_no_divs_falls_back_to_body_paragraphs() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc/></teiHeader>
        <text><body>
            <p>Loose paragraph one.</p>
            <p>Loose paragraph two.</p>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 1
    assert sections[0].title == "Body"
    assert "Loose paragraph one." in sections[0].text


def test_extract_sections_no_body_returns_abstract_only() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc><abstract>
            <p>Abstract.</p>
        </abstract></profileDesc></teiHeader>
        <text/>
    </TEI>"""
    sections = extract_sections(xml)
    assert len(sections) == 1
    assert sections[0].title == "Abstract"


def test_extract_sections_whitespace_in_paragraphs_normalized() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc/></teiHeader>
        <text><body>
            <div><head>Test</head><p>  lots   of   spaces  </p></div>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert sections[0].text == "lots of spaces"


def test_extract_sections_multiple_paragraphs_joined() -> None:
    xml = f"""<TEI {TEI_NS}>
        <teiHeader><profileDesc/></teiHeader>
        <text><body>
            <div><head>Multi</head>
                <p>Para one.</p>
                <p>Para two.</p>
            </div>
        </body></text>
    </TEI>"""
    sections = extract_sections(xml)
    assert "Para one." in sections[0].text
    assert "Para two." in sections[0].text
    assert "\n\n" in sections[0].text


# ===========================================================================
# GROBID client — pdf_to_tei
# ===========================================================================


def test_pdf_to_tei_calls_http_retry(monkeypatch: object, tmp_path: Path) -> None:
    from cv_rag.ingest import grobid_client

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake content")

    mock_response = MagicMock()
    mock_response.text = "<TEI>parsed</TEI>"

    captured_calls: list[dict] = []

    def fake_http_retry(client, method, url, **kwargs):  # type: ignore[no-untyped-def]
        captured_calls.append({"method": method, "url": url, "kwargs": kwargs})
        return mock_response

    monkeypatch.setattr(grobid_client, "http_request_with_retry", fake_http_retry)  # type: ignore[attr-defined]

    result = grobid_client.pdf_to_tei(pdf_file, "http://localhost:8070")

    assert result == "<TEI>parsed</TEI>"
    assert len(captured_calls) == 1
    assert captured_calls[0]["method"] == "POST"
    assert captured_calls[0]["url"] == "http://localhost:8070/api/processFulltextDocument"


def test_pdf_to_tei_strips_trailing_slash(monkeypatch: object, tmp_path: Path) -> None:
    from cv_rag.ingest import grobid_client

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 fake")

    mock_response = MagicMock()
    mock_response.text = "<TEI/>"

    captured_urls: list[str] = []

    def fake_http_retry(client, method, url, **kwargs):  # type: ignore[no-untyped-def]
        captured_urls.append(url)
        return mock_response

    monkeypatch.setattr(grobid_client, "http_request_with_retry", fake_http_retry)  # type: ignore[attr-defined]

    grobid_client.pdf_to_tei(pdf_file, "http://localhost:8070///")

    assert captured_urls[0] == "http://localhost:8070/api/processFulltextDocument"


def test_pdf_to_tei_prepare_kwargs_reopens_file(monkeypatch: object, tmp_path: Path) -> None:
    from cv_rag.ingest import grobid_client

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4 content")

    mock_response = MagicMock()
    mock_response.text = "<TEI/>"

    prepare_callbacks: list[object] = []

    def fake_http_retry(client, method, url, **kwargs):  # type: ignore[no-untyped-def]
        prepare = kwargs.get("prepare_kwargs")
        if prepare:
            # Call it twice to verify it can be called multiple times (retries)
            result1 = prepare()
            result2 = prepare()
            prepare_callbacks.extend([result1, result2])
            # Verify each call produces a valid file dict
            assert "files" in result1
            assert "files" in result2
        return mock_response

    monkeypatch.setattr(grobid_client, "http_request_with_retry", fake_http_retry)  # type: ignore[attr-defined]

    grobid_client.pdf_to_tei(pdf_file, "http://localhost:8070")

    assert len(prepare_callbacks) == 2
