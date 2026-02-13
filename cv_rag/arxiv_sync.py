from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from pydantic import BaseModel


ARXIV_VERSION_RE = re.compile(r"v\d+$")


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


def normalize_arxiv_id(raw: str) -> str:
    value = raw.strip()
    if value.lower().startswith("arxiv:"):
        value = value.split(":", 1)[1]

    if value.startswith("http://") or value.startswith("https://"):
        parsed = urlparse(value)
        path = parsed.path
        if path.startswith("/abs/"):
            value = path[len("/abs/") :]
        elif path.startswith("/pdf/"):
            value = path[len("/pdf/") :]
        else:
            value = path.strip("/")

    if value.endswith(".pdf"):
        value = value[: -len(".pdf")]

    value = value.strip("/")
    value = ARXIV_VERSION_RE.sub("", value)
    return value


def extract_version(raw: str) -> str | None:
    value = raw.strip()
    if value.lower().startswith("arxiv:"):
        value = value.split(":", 1)[1]
    if value.startswith("http://") or value.startswith("https://"):
        path = urlparse(value).path
        value = path.strip("/").split("/")[-1]
    if value.endswith(".pdf"):
        value = value[: -len(".pdf")]
    match = re.search(r"(v\d+)$", value)
    return match.group(1) if match else None


def _choose_pdf_url(entry: Any) -> str:
    for link in getattr(entry, "links", []):
        href = getattr(link, "href", "")
        link_type = getattr(link, "type", "")
        title = getattr(link, "title", "")
        if link_type == "application/pdf" or title == "pdf":
            return href

    entry_id = getattr(entry, "id", "")
    if "/abs/" in entry_id:
        return entry_id.replace("/abs/", "/pdf/") + ".pdf"
    return entry_id


def fetch_cs_cv_papers(
    limit: int,
    arxiv_api_url: str,
    timeout_seconds: float,
    user_agent: str,
) -> list[PaperMetadata]:
    params = {
        "search_query": "cat:cs.CV",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": 0,
        "max_results": limit,
    }
    headers = {"User-Agent": user_agent}

    with httpx.Client(timeout=timeout_seconds, headers=headers) as client:
        response = client.get(arxiv_api_url, params=params)
        response.raise_for_status()

    feed = feedparser.parse(response.text)
    papers: list[PaperMetadata] = []
    for entry in feed.entries:
        raw_id = str(getattr(entry, "id", "")).strip()
        if not raw_id:
            continue

        normalized = normalize_arxiv_id(raw_id)
        version = extract_version(raw_id)
        versioned = f"{normalized}{version or ''}"
        authors = [str(getattr(a, "name", "")) for a in getattr(entry, "authors", []) if getattr(a, "name", "")]

        papers.append(
            PaperMetadata(
                arxiv_id=normalized,
                arxiv_id_with_version=versioned,
                version=version,
                title=" ".join(str(getattr(entry, "title", "")).split()),
                summary=" ".join(str(getattr(entry, "summary", "")).split()),
                published=getattr(entry, "published", None),
                updated=getattr(entry, "updated", None),
                authors=authors,
                pdf_url=_choose_pdf_url(entry),
                abs_url=raw_id,
            )
        )
    return papers


def download_pdf(
    paper: PaperMetadata,
    pdf_dir: Path,
    timeout_seconds: float,
    user_agent: str,
    overwrite: bool = False,
) -> Path:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = pdf_dir / f"{paper.safe_file_stem()}.pdf"
    if out_path.exists() and not overwrite:
        return out_path

    headers = {"User-Agent": user_agent}
    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        with client.stream("GET", paper.pdf_url) as response:
            response.raise_for_status()
            with out_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)

    return out_path


def write_metadata_json(papers: list[PaperMetadata], metadata_json_path: Path) -> None:
    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [paper.model_dump() for paper in papers]
    metadata_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
