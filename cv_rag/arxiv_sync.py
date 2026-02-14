from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import feedparser
import httpx
from pydantic import BaseModel

from cv_rag.http_retry import http_request_with_retry

ARXIV_VERSION_RE = re.compile(r"v\d+$")
ARXIV_RSS_URL = "https://export.arxiv.org/rss/cs.CV"
ARXIV_ID_BATCH_SIZE = 50


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
    entry_link = getattr(entry, "link", "")
    if "/abs/" in entry_link:
        return entry_link.replace("/abs/", "/pdf/") + ".pdf"
    return entry_id


def _clean_rss_summary(summary: str) -> str:
    compact = " ".join(summary.split())
    marker = "Abstract:"
    marker_index = compact.find(marker)
    if marker_index >= 0:
        return compact[marker_index + len(marker) :].strip()
    return compact


def _extract_rss_authors(entry: Any) -> list[str]:
    raw_author = str(getattr(entry, "author", "")).strip()
    if not raw_author:
        raw_author = ", ".join(
            str(getattr(author, "name", "")).strip()
            for author in getattr(entry, "authors", [])
            if str(getattr(author, "name", "")).strip()
        )
    if not raw_author:
        return []
    return [author.strip() for author in raw_author.split(",") if author.strip()]


def _fetch_arxiv_api_feed(
    start: int,
    max_results: int,
    arxiv_api_url: str,
    timeout_seconds: float,
    user_agent: str,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
) -> str:
    params = {
        "search_query": "cat:cs.CV",
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": max_results,
    }
    headers = {"User-Agent": user_agent}

    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        response = http_request_with_retry(
            client, "GET", arxiv_api_url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label="arXiv request",
            params=params,
        )
        return response.text


def _fetch_arxiv_id_feed(
    arxiv_api_url: str,
    ids: list[str],
    timeout_seconds: float,
    user_agent: str,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
) -> str:
    if not ids:
        return ""

    params = {"id_list": ",".join(ids)}
    headers = {"User-Agent": user_agent}

    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        response = http_request_with_retry(
            client, "GET", arxiv_api_url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label="arXiv ID request",
            params=params,
        )
        return response.text


def _fetch_arxiv_rss_feed(timeout_seconds: float, user_agent: str) -> str:
    headers = {"User-Agent": user_agent}
    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        response = client.get(ARXIV_RSS_URL)
        response.raise_for_status()
    return response.text


def _parse_api_feed(feed_text: str) -> list[PaperMetadata]:
    feed = feedparser.parse(feed_text)
    papers: list[PaperMetadata] = []
    for entry in feed.entries:
        raw_id = str(getattr(entry, "id", "")).strip()
        if not raw_id:
            continue

        normalized = normalize_arxiv_id(raw_id)
        version = extract_version(raw_id)
        versioned = f"{normalized}{version or ''}"
        authors = [
            str(getattr(a, "name", ""))
            for a in getattr(entry, "authors", [])
            if getattr(a, "name", "")
        ]

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


def _parse_rss_feed(feed_text: str, limit: int) -> list[PaperMetadata]:
    feed = feedparser.parse(feed_text)
    papers: list[PaperMetadata] = []
    for entry in feed.entries[:limit]:
        abs_url = str(getattr(entry, "link", "")).strip()
        if not abs_url:
            continue

        raw_id = str(getattr(entry, "id", "")).strip()
        normalized = normalize_arxiv_id(abs_url)
        if not normalized:
            continue
        version = extract_version(raw_id)
        versioned = f"{normalized}{version or ''}"
        pdf_url = (
            abs_url.replace("/abs/", "/pdf/") + ".pdf"
            if "/abs/" in abs_url
            else f"https://arxiv.org/pdf/{normalized}.pdf"
        )

        papers.append(
            PaperMetadata(
                arxiv_id=normalized,
                arxiv_id_with_version=versioned,
                version=version,
                title=" ".join(str(getattr(entry, "title", "")).split()),
                summary=_clean_rss_summary(str(getattr(entry, "summary", ""))),
                published=getattr(entry, "published", None),
                updated=getattr(entry, "updated", None),
                authors=_extract_rss_authors(entry),
                pdf_url=pdf_url,
                abs_url=abs_url,
            )
        )
    return papers


def _append_filtered_papers(
    *,
    candidates: list[PaperMetadata],
    selected: list[PaperMetadata],
    selected_limit: int,
    skip_versions: set[str],
    seen_versions: set[str],
    scan_budget: int,
) -> tuple[int, int]:
    scanned = 0
    skipped = 0
    if scan_budget <= 0:
        return scanned, skipped

    for paper in candidates:
        if scanned >= scan_budget or len(selected) >= selected_limit:
            break
        scanned += 1

        versioned = paper.arxiv_id_with_version.strip()
        if versioned:
            if versioned in seen_versions:
                continue
            seen_versions.add(versioned)
            if versioned in skip_versions:
                skipped += 1
                continue
        elif "" in skip_versions:
            skipped += 1
            continue

        selected.append(paper)

    return scanned, skipped


def fetch_cs_cv_papers(
    limit: int,
    arxiv_api_url: str,
    timeout_seconds: float,
    user_agent: str,
    max_retries: int = 5,
    backoff_start_seconds: float = 2.0,
    backoff_cap_seconds: float = 30.0,
    skip_arxiv_id_with_version: set[str] | None = None,
    max_scan_results: int | None = None,
    stats: dict[str, int] | None = None,
) -> list[PaperMetadata]:
    if limit <= 0:
        return []

    selected: list[PaperMetadata] = []
    seen_versions: set[str] = set()
    skip_versions = {item.strip() for item in (skip_arxiv_id_with_version or set()) if item.strip()}
    scan_cap = max_scan_results if max_scan_results is not None else max(limit * 10, limit)
    scan_cap = max(scan_cap, limit)
    scanned_total = 0
    skipped_total = 0
    page_start = 0

    try:
        while len(selected) < limit and scanned_total < scan_cap:
            remaining_scan_budget = scan_cap - scanned_total
            batch_size = min(limit, remaining_scan_budget)
            if batch_size <= 0:
                break

            api_feed = _fetch_arxiv_api_feed(
                start=page_start,
                max_results=batch_size,
                arxiv_api_url=arxiv_api_url,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
                max_retries=max_retries,
                backoff_start_seconds=backoff_start_seconds,
                backoff_cap_seconds=backoff_cap_seconds,
            )
            candidates = _parse_api_feed(api_feed)
            if not candidates:
                break

            scanned_batch, skipped_batch = _append_filtered_papers(
                candidates=candidates,
                selected=selected,
                selected_limit=limit,
                skip_versions=skip_versions,
                seen_versions=seen_versions,
                scan_budget=remaining_scan_budget,
            )
            scanned_total += scanned_batch
            skipped_total += skipped_batch
            page_start += len(candidates)
            if len(candidates) < batch_size:
                break
    except httpx.HTTPStatusError as exc:
        if exc.response is None or exc.response.status_code != 429:
            raise
        rss_feed = _fetch_arxiv_rss_feed(timeout_seconds=timeout_seconds, user_agent=user_agent)
        rss_candidates = _parse_rss_feed(rss_feed, limit=scan_cap)
        remaining_scan_budget = scan_cap - scanned_total
        scanned_batch, skipped_batch = _append_filtered_papers(
            candidates=rss_candidates,
            selected=selected,
            selected_limit=limit,
            skip_versions=skip_versions,
            seen_versions=seen_versions,
            scan_budget=remaining_scan_budget,
        )
        scanned_total += scanned_batch
        skipped_total += skipped_batch

    if stats is not None:
        stats.clear()
        stats.update(
            {
                "requested": limit,
                "selected": len(selected),
                "scanned": scanned_total,
                "skipped": skipped_total,
            }
        )
    return selected


def _canonical_requested_id(raw_id: str) -> str:
    normalized = normalize_arxiv_id(raw_id)
    if not normalized:
        return ""
    version = extract_version(raw_id)
    return f"{normalized}{version or ''}"


def _iter_id_batches(ids: list[str], batch_size: int) -> list[list[str]]:
    safe_batch_size = max(1, batch_size)
    return [ids[idx : idx + safe_batch_size] for idx in range(0, len(ids), safe_batch_size)]


def fetch_papers_by_ids(
    ids: list[str],
    arxiv_api_url: str,
    timeout_seconds: float,
    user_agent: str,
    max_retries: int = 5,
    backoff_start_seconds: float = 2.0,
    backoff_cap_seconds: float = 30.0,
    id_batch_size: int = ARXIV_ID_BATCH_SIZE,
) -> list[PaperMetadata]:
    requested_ids: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        cleaned = _canonical_requested_id(raw)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        requested_ids.append(cleaned)

    if not requested_ids:
        return []

    fetched: list[PaperMetadata] = []
    for id_batch in _iter_id_batches(requested_ids, batch_size=id_batch_size):
        try:
            api_feed = _fetch_arxiv_id_feed(
                arxiv_api_url=arxiv_api_url,
                ids=id_batch,
                timeout_seconds=timeout_seconds,
                user_agent=user_agent,
                max_retries=max_retries,
                backoff_start_seconds=backoff_start_seconds,
                backoff_cap_seconds=backoff_cap_seconds,
            )
        except (httpx.HTTPError, httpx.HTTPStatusError, RuntimeError):
            continue

        if api_feed:
            fetched.extend(_parse_api_feed(api_feed))

    by_versioned: dict[str, PaperMetadata] = {}
    by_base: dict[str, PaperMetadata] = {}
    for paper in fetched:
        if paper.arxiv_id_with_version:
            by_versioned[paper.arxiv_id_with_version] = paper
        by_base[paper.arxiv_id] = paper

    papers: list[PaperMetadata] = []
    for requested_id in requested_ids:
        base_id = normalize_arxiv_id(requested_id)
        requested_version = extract_version(requested_id)
        download_id = requested_id if requested_version else base_id

        source = by_versioned.get(download_id) or by_base.get(base_id)
        title = source.title if source and source.title else f"arXiv:{download_id}"
        summary = source.summary if source else ""
        authors = source.authors if source else []
        published = source.published if source else None
        updated = source.updated if source else None

        papers.append(
            PaperMetadata(
                arxiv_id=base_id,
                arxiv_id_with_version=download_id,
                version=requested_version or (source.version if source else None),
                title=title,
                summary=summary,
                published=published,
                updated=updated,
                authors=authors,
                pdf_url=f"https://arxiv.org/pdf/{download_id}.pdf",
                abs_url=f"https://arxiv.org/abs/{download_id}",
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
    with (
        httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client,
        client.stream("GET", paper.pdf_url) as response,
    ):
            response.raise_for_status()
            with out_path.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)

    return out_path


def write_metadata_json(papers: list[PaperMetadata], metadata_json_path: Path) -> None:
    metadata_json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [paper.model_dump() for paper in papers]
    metadata_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
