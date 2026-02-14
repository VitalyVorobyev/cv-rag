from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx

from cv_rag.http_retry import http_request_with_retry

logger = logging.getLogger(__name__)

DEFAULT_TIER_A_SEED_PATH = Path("data/curation/tierA_seed.txt")
README_CANDIDATES = ("README.md", "readme.md")
_REPO_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_ARXIV_BASE_ID_RE = r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})"
_ARXIV_ABS_URL_RE = re.compile(
    rf"https?://arxiv\.org/abs/(?P<base_id>{_ARXIV_BASE_ID_RE})(?P<version>v\d+)?\b",
    flags=re.IGNORECASE,
)
_ARXIV_PDF_URL_RE = re.compile(
    rf"https?://arxiv\.org/pdf/(?P<base_id>{_ARXIV_BASE_ID_RE})(?P<version>v\d+)?\.pdf\b",
    flags=re.IGNORECASE,
)
_ARXIV_TEXT_RE = re.compile(
    rf"\barxiv(?:\s*:\s*|\s+)(?P<base_id>{_ARXIV_BASE_ID_RE})(?P<version>v\d+)?\b",
    flags=re.IGNORECASE,
)


@dataclass(slots=True, frozen=True)
class RepoSource:
    repo: str
    source_url: str


@dataclass(slots=True, frozen=True)
class ArxivMatch:
    base_id: str
    arxiv_id: str
    version: int | None
    raw_match: str


@dataclass(slots=True, frozen=True)
class AwesomeSeedRecord:
    base_id: str
    arxiv_id: str
    source_repo: str
    source_url: str
    found_in: str
    raw_match: str

    def to_dict(self) -> dict[str, str]:
        return {
            "base_id": self.base_id,
            "arxiv_id": self.arxiv_id,
            "source_repo": self.source_repo,
            "source_url": self.source_url,
            "found_in": self.found_in,
            "raw_match": self.raw_match,
        }


@dataclass(slots=True)
class AwesomeSeedStats:
    repos_processed: int
    total_matches: int
    unique_ids: int
    repo_counts: dict[str, int]
    jsonl_path: Path
    tier_a_seed_path: Path

    def top_repos(self, limit: int = 10) -> list[tuple[str, int]]:
        if limit <= 0:
            return []
        return sorted(self.repo_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


@dataclass(slots=True, frozen=True)
class _FetchedContent:
    text: str
    found_in: str


def parse_repo_source(value: str) -> RepoSource | None:
    stripped = value.strip()
    if not stripped or stripped.startswith("#"):
        return None

    owner = ""
    repo = ""

    if stripped.startswith("http://") or stripped.startswith("https://"):
        parsed = urlparse(stripped)
        if parsed.netloc.casefold() not in {"github.com", "www.github.com"}:
            raise ValueError(f"Unsupported repo URL: {stripped}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 2:
            raise ValueError(f"Expected GitHub URL in owner/repo form: {stripped}")
        owner, repo = parts[0], parts[1]
    else:
        parts = stripped.split("/")
        if len(parts) != 2:
            raise ValueError(f"Expected owner/repo entry, got: {stripped}")
        owner, repo = parts[0].strip(), parts[1].strip()

    repo = repo.removesuffix(".git")
    if not owner or not repo:
        raise ValueError(f"Invalid GitHub repo reference: {stripped}")
    if not _REPO_PART_RE.fullmatch(owner) or not _REPO_PART_RE.fullmatch(repo):
        raise ValueError(f"Invalid GitHub repo reference: {stripped}")

    normalized = f"{owner}/{repo}"
    return RepoSource(repo=normalized, source_url=f"https://github.com/{normalized}")


def load_repo_sources(path: Path) -> list[RepoSource]:
    if not path.exists():
        raise FileNotFoundError(f"Sources file not found: {path}")

    repos: list[RepoSource] = []
    seen: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            repo_source = parse_repo_source(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
        if repo_source is None or repo_source.repo in seen:
            continue
        seen.add(repo_source.repo)
        repos.append(repo_source)
    return repos


def _normalize_base_id(base_id: str) -> str:
    cleaned = base_id.strip().rstrip(")]}>,.;:")
    return cleaned.casefold() if "/" in cleaned else cleaned


def _parse_version(version: str | None) -> int | None:
    if not version:
        return None
    return int(version[1:])


def _build_arxiv_id(base_id: str, version: int | None) -> str:
    return f"{base_id}v{version}" if version is not None else base_id


def extract_arxiv_matches(text: str) -> list[ArxivMatch]:
    if not text:
        return []

    matches_with_pos: list[tuple[int, ArxivMatch]] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for pattern in (_ARXIV_ABS_URL_RE, _ARXIV_PDF_URL_RE, _ARXIV_TEXT_RE):
        for match in pattern.finditer(text):
            raw_match = match.group(0)
            base_id = _normalize_base_id(match.group("base_id"))
            version = _parse_version(match.group("version"))
            arxiv_id = _build_arxiv_id(base_id, version)
            span_key = (match.start(), match.end(), arxiv_id)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            matches_with_pos.append(
                (
                    match.start(),
                    ArxivMatch(
                        base_id=base_id,
                        arxiv_id=arxiv_id,
                        version=version,
                        raw_match=raw_match,
                    ),
                )
            )

    matches_with_pos.sort(key=lambda item: item[0])
    return [item[1] for item in matches_with_pos]


def _request_optional_text(
    *,
    client: httpx.Client,
    url: str,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
    allow_404: bool,
) -> str | None:
    try:
        response = http_request_with_retry(
            client,
            "GET",
            url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label=f"GitHub request: {url}",
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return None
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if allow_404 and status_code == 404:
            return None
        logger.warning("GitHub request failed (%s): %s", status_code, url)
        return None
    return response.text


def _fetch_repo_readme(
    *,
    client: httpx.Client,
    repo: str,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
) -> _FetchedContent | None:
    repo_url = f"https://github.com/{repo}"
    for readme_name in README_CANDIDATES:
        raw_url = f"{repo_url}/raw/HEAD/{readme_name}"
        text = _request_optional_text(
            client=client,
            url=raw_url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            allow_404=True,
        )
        if text is not None:
            return _FetchedContent(text=text, found_in=readme_name)

    html_fallback = _request_optional_text(
        client=client,
        url=repo_url,
        max_retries=max_retries,
        backoff_start_seconds=backoff_start_seconds,
        backoff_cap_seconds=backoff_cap_seconds,
        allow_404=True,
    )
    if html_fallback is None:
        return None
    return _FetchedContent(text=html_fallback, found_in="GitHub HTML")


def seed_awesome_sources(
    *,
    sources_path: Path,
    out_dir: Path,
    user_agent: str,
    timeout_seconds: float = 20.0,
    max_retries: int = 5,
    backoff_start_seconds: float = 1.0,
    backoff_cap_seconds: float = 30.0,
    delay_seconds: float = 0.2,
    tier_a_seed_path: Path = DEFAULT_TIER_A_SEED_PATH,
) -> AwesomeSeedStats:
    repo_sources = load_repo_sources(sources_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "awesome_seed.jsonl"

    tier_a_seed_path.parent.mkdir(parents=True, exist_ok=True)
    unique_base_ids: set[str] = set()
    repo_counts: Counter[str] = Counter()
    total_matches = 0

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/plain, text/markdown;q=0.9, text/html;q=0.8",
    }

    with (
        httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        for repo_source in repo_sources:
            fetched = _fetch_repo_readme(
                client=client,
                repo=repo_source.repo,
                max_retries=max_retries,
                backoff_start_seconds=backoff_start_seconds,
                backoff_cap_seconds=backoff_cap_seconds,
            )
            if fetched is None:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                continue

            matches = extract_arxiv_matches(fetched.text)
            if matches:
                repo_counts[repo_source.repo] += len(matches)

            for match in matches:
                unique_base_ids.add(match.base_id)
                total_matches += 1
                record = AwesomeSeedRecord(
                    base_id=match.base_id,
                    arxiv_id=match.arxiv_id,
                    source_repo=repo_source.repo,
                    source_url=repo_source.source_url,
                    found_in=fetched.found_in,
                    raw_match=match.raw_match,
                )
                jsonl_file.write(json.dumps(record.to_dict(), ensure_ascii=False))
                jsonl_file.write("\n")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    sorted_ids = sorted(unique_base_ids)
    with tier_a_seed_path.open("w", encoding="utf-8") as tier_file:
        if sorted_ids:
            tier_file.write("\n".join(sorted_ids))
            tier_file.write("\n")

    return AwesomeSeedStats(
        repos_processed=len(repo_sources),
        total_matches=total_matches,
        unique_ids=len(unique_base_ids),
        repo_counts=dict(repo_counts),
        jsonl_path=jsonl_path,
        tier_a_seed_path=tier_a_seed_path,
    )
