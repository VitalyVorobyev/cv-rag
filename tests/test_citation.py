from cv_rag.retrieval.hybrid import format_citation


def test_format_citation() -> None:
    assert format_citation("2401.12345", "Methods") == "arXiv:2401.12345 §Methods"


def test_format_citation_untitled_section() -> None:
    assert format_citation("2401.12345", "") == "arXiv:2401.12345 §Untitled"
