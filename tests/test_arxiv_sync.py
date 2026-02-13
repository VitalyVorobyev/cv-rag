from cv_rag.arxiv_sync import normalize_arxiv_id


def test_normalize_arxiv_id_modern_abs_url() -> None:
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"


def test_normalize_arxiv_id_prefixed() -> None:
    assert normalize_arxiv_id("arXiv:2401.12345v1") == "2401.12345"


def test_normalize_arxiv_id_legacy_pdf_url() -> None:
    assert normalize_arxiv_id("https://arxiv.org/pdf/cs/9901001v2.pdf") == "cs/9901001"
