from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import httpx

from cv_rag.ingest.arxiv_client import normalize_arxiv_id
from cv_rag.seeding.awesome import extract_doi_matches
from cv_rag.seeding.doi import normalize_doi
from cv_rag.shared.http import http_request_with_retry

logger = logging.getLogger(__name__)

DEFAULT_VISIONBIB_OUT_DIR = Path("data/curation/visionbib")
DEFAULT_TIER_A_VISIONBIB_DOIS_PATH = Path("data/curation/tierA_dois_visionbib.txt")
DEFAULT_TIER_A_VISIONBIB_URLS_PATH = Path("data/curation/tierA_urls_visionbib.txt")
DEFAULT_TIER_A_VISIONBIB_ARXIV_PATH = Path("data/curation/tierA_arxiv_visionbib.txt")

_PREFIX_LINE_RE = re.compile(r"^https?://", flags=re.IGNORECASE)
_RANGE_TOKEN_RE = re.compile(r"^(?P<stem>[A-Za-z][A-Za-z_-]*)(?P<idx>\d+)$")
_PDF_LINK_RE = re.compile(r"\.pdf$", flags=re.IGNORECASE)


@dataclass(slots=True, frozen=True)
class VisionBibPageRange:
    stem: str
    start: int
    end: int


@dataclass(slots=True, frozen=True)
class VisionBibSourceSpec:
    prefix_url: str
    ranges: list[VisionBibPageRange]


@dataclass(slots=True, frozen=True)
class PageTarget:
    stem: str
    index: int
    page_url: str


@dataclass(slots=True, frozen=True)
class VisionBibLinkRecord:
    kind: str  # doi|pdf|arxiv
    normalized_value: str
    source_page_url: str
    raw_link: str
    context: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "normalized_value": self.normalized_value,
            "source_page_url": self.source_page_url,
            "raw_link": self.raw_link,
            "context": self.context,
        }


@dataclass(slots=True)
class VisionBibSeedStats:
    pages_requested: int
    pages_succeeded: int
    pages_failed: int
    total_doi_matches: int
    total_pdf_matches: int
    total_arxiv_matches: int
    unique_dois: int
    unique_pdf_urls: int
    unique_arxiv_ids: int
    doi_jsonl_path: Path
    url_jsonl_path: Path
    arxiv_jsonl_path: Path
    tier_a_dois_path: Path
    tier_a_urls_path: Path
    tier_a_arxiv_path: Path
    page_counts: dict[str, int]

    def top_pages(self, limit: int = 10) -> list[tuple[str, int]]:
        if limit <= 0:
            return []
        return sorted(self.page_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


class _HrefExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() != "a":
            return
        for key, value in attrs:
            if key.casefold() != "href":
                continue
            if not value:
                continue
            self.hrefs.append(value)


def _parse_compact_range(start_token: str, end_token: str) -> VisionBibPageRange:
    start_match = _RANGE_TOKEN_RE.fullmatch(start_token)
    end_match = _RANGE_TOKEN_RE.fullmatch(end_token)
    if start_match is None or end_match is None:
        raise ValueError(f"Expected compact range tokens like 'compute42 compute114', got: {start_token} {end_token}")

    stem_start = start_match.group("stem")
    stem_end = end_match.group("stem")
    if stem_start != stem_end:
        raise ValueError(f"Range stems do not match: {start_token} {end_token}")

    start_idx = int(start_match.group("idx"))
    end_idx = int(end_match.group("idx"))
    if start_idx > end_idx:
        raise ValueError(f"Range start must be <= end: {start_token} {end_token}")
    return VisionBibPageRange(stem=stem_start, start=start_idx, end=end_idx)


def load_visionbib_sources(path: Path) -> VisionBibSourceSpec:
    if not path.exists():
        raise FileNotFoundError(f"VisionBib sources file not found: {path}")

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError(f"No VisionBib sources found in: {path}")

    prefix = lines[0]
    if _PREFIX_LINE_RE.search(prefix) is None:
        raise ValueError(f"First non-comment line must be a URL prefix: {prefix}")

    ranges: list[VisionBibPageRange] = []
    for line_no, line in enumerate(lines[1:], start=2):
        parts = line.split()
        if len(parts) == 3:
            stem, start_text, end_text = parts
            if not start_text.isdigit() or not end_text.isdigit():
                raise ValueError(f"{path}:{line_no}: expected numeric start/end in '{line}'")
            start_idx = int(start_text)
            end_idx = int(end_text)
            if start_idx > end_idx:
                raise ValueError(f"{path}:{line_no}: start must be <= end in '{line}'")
            ranges.append(VisionBibPageRange(stem=stem, start=start_idx, end=end_idx))
            continue

        if len(parts) == 2:
            try:
                ranges.append(_parse_compact_range(parts[0], parts[1]))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            continue

        raise ValueError(
            f"{path}:{line_no}: expected 'stem start end' or 'stemStart stemEnd', got: {line}"
        )

    if not ranges:
        raise ValueError(f"No ranges found in VisionBib sources: {path}")

    return VisionBibSourceSpec(prefix_url=prefix, ranges=ranges)


def expand_page_urls(spec: VisionBibSourceSpec) -> list[PageTarget]:
    prefix = spec.prefix_url.rstrip("/")
    pages: list[PageTarget] = []
    seen_urls: set[str] = set()
    for page_range in spec.ranges:
        for index in range(page_range.start, page_range.end + 1):
            page_url = f"{prefix}/{page_range.stem}{index}.html"
            if page_url in seen_urls:
                continue
            seen_urls.add(page_url)
            pages.append(PageTarget(stem=page_range.stem, index=index, page_url=page_url))
    return pages


def _extract_page_hrefs(page_html: str, page_url: str) -> list[str]:
    parser = _HrefExtractor()
    parser.feed(page_html)

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_href in parser.hrefs:
        href = raw_href.strip()
        if not href:
            continue
        lowered = href.casefold()
        if lowered.startswith("javascript:") or lowered.startswith("mailto:"):
            continue
        absolute = urljoin(page_url, href)
        parsed = urlparse(absolute)
        if parsed.scheme.casefold() not in {"http", "https"}:
            continue
        normalized_url = parsed._replace(fragment="").geturl()
        if normalized_url in seen:
            continue
        seen.add(normalized_url)
        normalized.append(normalized_url)
    return normalized


def _classify_link(url: str) -> tuple[str | None, str | None]:
    parsed = urlparse(url)
    host = parsed.netloc.casefold()

    if host.endswith("doi.org"):
        doi = normalize_doi(unquote(parsed.path.lstrip("/")))
        if doi:
            return "doi", doi

    path_no_fragment = parsed.path or ""
    if _PDF_LINK_RE.search(path_no_fragment):
        return "pdf", parsed._replace(fragment="").geturl()

    if host.endswith("arxiv.org") and (path_no_fragment.startswith("/abs/") or path_no_fragment.startswith("/pdf/")):
        arxiv_id = normalize_arxiv_id(url)
        if arxiv_id:
            return "arxiv", arxiv_id

    return None, None


def _request_optional_page(
    *,
    client: httpx.Client,
    url: str,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
) -> str | None:
    try:
        response = http_request_with_retry(
            client,
            "GET",
            url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label=f"VisionBib request: {url}",
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        logger.warning("VisionBib request failed (%s): %s", status, url)
        return None
    return response.text


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        if lines:
            file_handle.write("\n".join(lines))
            file_handle.write("\n")


def _write_records(path: Path, records: list[VisionBibLinkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            file_handle.write("\n")


def seed_visionbib_sources(
    *,
    sources_path: Path,
    out_dir: Path = DEFAULT_VISIONBIB_OUT_DIR,
    user_agent: str,
    timeout_seconds: float = 20.0,
    max_retries: int = 5,
    backoff_start_seconds: float = 1.0,
    backoff_cap_seconds: float = 30.0,
    delay_seconds: float = 0.2,
    tier_a_dois_path: Path = DEFAULT_TIER_A_VISIONBIB_DOIS_PATH,
    tier_a_urls_path: Path = DEFAULT_TIER_A_VISIONBIB_URLS_PATH,
    tier_a_arxiv_path: Path = DEFAULT_TIER_A_VISIONBIB_ARXIV_PATH,
) -> VisionBibSeedStats:
    spec = load_visionbib_sources(sources_path)
    targets = expand_page_urls(spec)

    doi_records: list[VisionBibLinkRecord] = []
    url_records: list[VisionBibLinkRecord] = []
    arxiv_records: list[VisionBibLinkRecord] = []
    page_counts: Counter[str] = Counter()

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml",
    }

    pages_succeeded = 0
    pages_failed = 0

    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
        for idx, target in enumerate(targets):
            if idx > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)

            html = _request_optional_page(
                client=client,
                url=target.page_url,
                max_retries=max_retries,
                backoff_start_seconds=backoff_start_seconds,
                backoff_cap_seconds=backoff_cap_seconds,
            )
            if html is None:
                pages_failed += 1
                continue

            pages_succeeded += 1
            page_seen: set[tuple[str, str]] = set()

            for href in _extract_page_hrefs(html, target.page_url):
                kind, normalized_value = _classify_link(href)
                if kind is None or normalized_value is None:
                    continue
                dedupe_key = (kind, normalized_value)
                if dedupe_key in page_seen:
                    continue
                page_seen.add(dedupe_key)
                record = VisionBibLinkRecord(
                    kind=kind,
                    normalized_value=normalized_value,
                    source_page_url=target.page_url,
                    raw_link=href,
                    context="href",
                )
                if kind == "doi":
                    doi_records.append(record)
                elif kind == "pdf":
                    url_records.append(record)
                elif kind == "arxiv":
                    arxiv_records.append(record)
                page_counts[target.page_url] += 1

            for doi_match in extract_doi_matches(html):
                dedupe_key = ("doi", doi_match.doi)
                if dedupe_key in page_seen:
                    continue
                page_seen.add(dedupe_key)
                doi_records.append(
                    VisionBibLinkRecord(
                        kind="doi",
                        normalized_value=doi_match.doi,
                        source_page_url=target.page_url,
                        raw_link=doi_match.raw_match,
                        context="text",
                    )
                )
                page_counts[target.page_url] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    doi_jsonl_path = out_dir / "visionbib_seed_doi.jsonl"
    url_jsonl_path = out_dir / "visionbib_seed_url.jsonl"
    arxiv_jsonl_path = out_dir / "visionbib_seed_arxiv.jsonl"

    _write_records(doi_jsonl_path, doi_records)
    _write_records(url_jsonl_path, url_records)
    _write_records(arxiv_jsonl_path, arxiv_records)

    unique_dois = sorted({record.normalized_value for record in doi_records})
    unique_urls = sorted({record.normalized_value for record in url_records})
    unique_arxiv = sorted({record.normalized_value for record in arxiv_records})

    _write_lines(tier_a_dois_path, unique_dois)
    _write_lines(tier_a_urls_path, unique_urls)
    _write_lines(tier_a_arxiv_path, unique_arxiv)

    return VisionBibSeedStats(
        pages_requested=len(targets),
        pages_succeeded=pages_succeeded,
        pages_failed=pages_failed,
        total_doi_matches=len(doi_records),
        total_pdf_matches=len(url_records),
        total_arxiv_matches=len(arxiv_records),
        unique_dois=len(unique_dois),
        unique_pdf_urls=len(unique_urls),
        unique_arxiv_ids=len(unique_arxiv),
        doi_jsonl_path=doi_jsonl_path,
        url_jsonl_path=url_jsonl_path,
        arxiv_jsonl_path=arxiv_jsonl_path,
        tier_a_dois_path=tier_a_dois_path,
        tier_a_urls_path=tier_a_urls_path,
        tier_a_arxiv_path=tier_a_arxiv_path,
        page_counts=dict(page_counts),
    )


class SeedVisionBibService:
    def run(
        self,
        *,
        sources_path: Path,
        out_dir: Path = DEFAULT_VISIONBIB_OUT_DIR,
        user_agent: str,
        timeout_seconds: float = 20.0,
        max_retries: int = 5,
        backoff_start_seconds: float = 1.0,
        backoff_cap_seconds: float = 30.0,
        delay_seconds: float = 0.2,
        tier_a_dois_path: Path = DEFAULT_TIER_A_VISIONBIB_DOIS_PATH,
        tier_a_urls_path: Path = DEFAULT_TIER_A_VISIONBIB_URLS_PATH,
        tier_a_arxiv_path: Path = DEFAULT_TIER_A_VISIONBIB_ARXIV_PATH,
    ) -> VisionBibSeedStats:
        return seed_visionbib_sources(
            sources_path=sources_path,
            out_dir=out_dir,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            delay_seconds=delay_seconds,
            tier_a_dois_path=tier_a_dois_path,
            tier_a_urls_path=tier_a_urls_path,
            tier_a_arxiv_path=tier_a_arxiv_path,
        )
