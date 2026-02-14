from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class PaperRecord:
    arxiv_id: str
    arxiv_id_with_version: str
    version: str | None
    title: str
    summary: str
    published: str | None
    updated: str | None
    authors: list[str]
    pdf_url: str
    abs_url: str
    pdf_path: Path | None
    tei_path: Path | None
