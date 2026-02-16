from __future__ import annotations

from pydantic import BaseModel


class PaperMetadata(BaseModel):
    arxiv_id: str
    arxiv_id_with_version: str
    version: str | None = None
    doc_id: str | None = None
    provenance_kind: str | None = None
    title: str
    summary: str
    published: str | None = None
    updated: str | None = None
    authors: list[str]
    pdf_url: str
    abs_url: str

    def safe_file_stem(self) -> str:
        raw = (self.arxiv_id_with_version or self.doc_id or self.arxiv_id).strip()
        return (
            raw.replace("/", "_")
            .replace(":", "_")
            .replace("?", "_")
            .replace("&", "_")
        )

    def resolved_doc_id(self) -> str:
        if self.doc_id and self.doc_id.strip():
            return self.doc_id.strip()
        if self.arxiv_id_with_version and self.arxiv_id_with_version.strip():
            return f"axv:{self.arxiv_id_with_version.strip()}"
        if self.arxiv_id and self.arxiv_id.strip():
            return f"axv:{self.arxiv_id.strip()}"
        return "axv:unknown"
