from __future__ import annotations

from pydantic import BaseModel


class PaperMetadata(BaseModel):
    arxiv_id: str
    arxiv_id_with_version: str
    version: str | None = None
    title: str
    summary: str
    published: str | None = None
    updated: str | None = None
    authors: list[str]
    pdf_url: str
    abs_url: str

    def safe_file_stem(self) -> str:
        return self.arxiv_id_with_version.replace("/", "_")
