from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from cv_rag.seeding.doi import DOI_TRAILING_CHARS, normalize_doi
from cv_rag.shared.http import http_request_with_retry
from cv_rag.storage.repositories import ReferenceRecord
from cv_rag.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)

DEFAULT_TIER_A_SEED_PATH = Path("data/curation/tierA_seed.txt")
DEFAULT_TIER_A_ARXIV_PATH = Path("data/curation/tierA_arxiv.txt")
DEFAULT_TIER_A_DOIS_PATH = Path("data/curation/tierA_dois.txt")

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

_DOI_CORE_RE = r"10\.\d{4,9}/[-._;()/:A-Z0-9]+"
_DOI_URL_RE = re.compile(
    rf"https?://(?:dx\.)?doi\.org/(?P<doi>{_DOI_CORE_RE})",
    flags=re.IGNORECASE,
)
_DOI_PREFIX_RE = re.compile(
    rf"\bdoi\s*:\s*(?P<doi>{_DOI_CORE_RE})",
    flags=re.IGNORECASE,
)
_DOI_BARE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<doi>{_DOI_CORE_RE})",
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
class DoiMatch:
    doi: str
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


@dataclass(slots=True, frozen=True)
class AwesomeDoiSeedRecord:
    doi: str
    source_repo: str
    source_url: str
    found_in: str
    raw_match: str

    def to_dict(self) -> dict[str, str]:
        return {
            "doi": self.doi,
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
    total_doi_matches: int
    unique_dois: int
    doi_repo_counts: dict[str, int]
    doi_jsonl_path: Path
    tier_a_arxiv_path: Path
    tier_a_dois_path: Path
    run_id: str | None = None
    run_artifact_path: Path | None = None

    def top_repos(self, limit: int = 10) -> list[tuple[str, int]]:
        if limit <= 0:
            return []
        return sorted(self.repo_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def top_doi_repos(self, limit: int = 10) -> list[tuple[str, int]]:
        if limit <= 0:
            return []
        return sorted(self.doi_repo_counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


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


def _spans_overlap(start: int, end: int, spans: list[tuple[int, int, str]]) -> bool:
    return any(start < existing_end and end > existing_start for existing_start, existing_end, _ in spans)


def extract_doi_matches(text: str) -> list[DoiMatch]:
    if not text:
        return []

    matches_with_pos: list[tuple[int, DoiMatch]] = []
    selected_spans: list[tuple[int, int, str]] = []
    seen_spans: set[tuple[int, int, str]] = set()

    for pattern in (_DOI_URL_RE, _DOI_PREFIX_RE, _DOI_BARE_RE):
        for match in pattern.finditer(text):
            doi = normalize_doi(match.group("doi"))
            if not doi:
                continue

            raw_match = match.group(0)
            while raw_match and raw_match[-1] in DOI_TRAILING_CHARS:
                raw_match = raw_match[:-1]
            if not raw_match:
                continue

            start, end = match.span("doi")
            span_key = (start, end, doi)
            if span_key in seen_spans:
                continue
            if _spans_overlap(start, end, selected_spans):
                continue
            seen_spans.add(span_key)
            selected_spans.append(span_key)
            matches_with_pos.append((start, DoiMatch(doi=doi, raw_match=raw_match)))

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


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        if lines:
            file_handle.write("\n".join(lines))
            file_handle.write("\n")


def write_seed_outputs(
    *,
    arxiv_records: list[AwesomeSeedRecord],
    doi_records: list[AwesomeDoiSeedRecord],
    out_dir: Path,
    tier_a_seed_path: Path = DEFAULT_TIER_A_SEED_PATH,
    tier_a_arxiv_path: Path = DEFAULT_TIER_A_ARXIV_PATH,
    tier_a_dois_path: Path = DEFAULT_TIER_A_DOIS_PATH,
) -> tuple[Path, Path, Path, Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "awesome_seed.jsonl"
    doi_jsonl_path = out_dir / "awesome_seed_doi.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as file_handle:
        for record in arxiv_records:
            file_handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            file_handle.write("\n")

    with doi_jsonl_path.open("w", encoding="utf-8") as file_handle:
        for record in doi_records:
            file_handle.write(json.dumps(record.to_dict(), ensure_ascii=False))
            file_handle.write("\n")

    sorted_arxiv_ids = sorted({record.base_id for record in arxiv_records})
    sorted_dois = sorted({record.doi for record in doi_records})

    # Backward-compatible output.
    _write_lines(tier_a_seed_path, sorted_arxiv_ids)
    _write_lines(tier_a_arxiv_path, sorted_arxiv_ids)
    _write_lines(tier_a_dois_path, sorted_dois)
    return (
        jsonl_path,
        doi_jsonl_path,
        tier_a_seed_path,
        tier_a_arxiv_path,
        len(sorted_arxiv_ids),
        len(sorted_dois),
    )


def _reference_records_from_seed_records(
    *,
    arxiv_records: list[AwesomeSeedRecord],
    doi_records: list[AwesomeDoiSeedRecord],
    discovered_at_unix: int,
    source_kind: str,
) -> list[ReferenceRecord]:
    refs: list[ReferenceRecord] = []
    for item in arxiv_records:
        refs.append(
            ReferenceRecord(
                ref_type="arxiv",
                normalized_value=item.arxiv_id,
                source_kind=source_kind,
                source_ref=item.source_url,
                discovered_at_unix=discovered_at_unix,
            )
        )
    for item in doi_records:
        refs.append(
            ReferenceRecord(
                ref_type="doi",
                normalized_value=item.doi,
                source_kind=source_kind,
                source_ref=item.source_url,
                discovered_at_unix=discovered_at_unix,
            )
        )
    return refs


def _write_run_reference_artifact(
    *,
    runs_dir: Path,
    run_id: str,
    refs: list[ReferenceRecord],
) -> Path:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "awesome_references.jsonl"
    with path.open("w", encoding="utf-8") as file_handle:
        for ref in refs:
            file_handle.write(
                json.dumps(
                    {
                        "ref_type": ref.ref_type,
                        "normalized_value": ref.normalized_value,
                        "source_kind": ref.source_kind,
                        "source_ref": ref.source_ref,
                        "discovered_at_unix": ref.discovered_at_unix,
                    },
                    ensure_ascii=False,
                )
            )
            file_handle.write("\n")
    return path


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
    tier_a_arxiv_path: Path = DEFAULT_TIER_A_ARXIV_PATH,
    tier_a_dois_path: Path = DEFAULT_TIER_A_DOIS_PATH,
    run_id: str | None = None,
    runs_dir: Path | None = None,
    sqlite_path: Path | None = None,
    source_kind: str = "curated_repo",
    candidate_retry_days: int = 14,
    candidate_max_retries: int = 5,
) -> AwesomeSeedStats:
    repo_sources = load_repo_sources(sources_path)
    discovered_at_unix = int(time.time())
    effective_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_artifact_path: Path | None = None

    arxiv_records: list[AwesomeSeedRecord] = []
    doi_records: list[AwesomeDoiSeedRecord] = []
    repo_counts: Counter[str] = Counter()
    doi_repo_counts: Counter[str] = Counter()
    total_matches = 0
    total_doi_matches = 0

    headers = {
        "User-Agent": user_agent,
        "Accept": "text/plain, text/markdown;q=0.9, text/html;q=0.8",
    }

    with httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client:
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

            arxiv_matches = extract_arxiv_matches(fetched.text)
            doi_matches = extract_doi_matches(fetched.text)
            if arxiv_matches:
                repo_counts[repo_source.repo] += len(arxiv_matches)
            if doi_matches:
                doi_repo_counts[repo_source.repo] += len(doi_matches)

            for match in arxiv_matches:
                total_matches += 1
                arxiv_records.append(
                    AwesomeSeedRecord(
                        base_id=match.base_id,
                        arxiv_id=match.arxiv_id,
                        source_repo=repo_source.repo,
                        source_url=repo_source.source_url,
                        found_in=fetched.found_in,
                        raw_match=match.raw_match,
                    )
                )

            for match in doi_matches:
                total_doi_matches += 1
                doi_records.append(
                    AwesomeDoiSeedRecord(
                        doi=match.doi,
                        source_repo=repo_source.repo,
                        source_url=repo_source.source_url,
                        found_in=fetched.found_in,
                        raw_match=match.raw_match,
                    )
                )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    (
        jsonl_path,
        doi_jsonl_path,
        tier_a_seed_written_path,
        tier_a_arxiv_written_path,
        unique_arxiv_count,
        unique_doi_count,
    ) = write_seed_outputs(
        arxiv_records=arxiv_records,
        doi_records=doi_records,
        out_dir=out_dir,
        tier_a_seed_path=tier_a_seed_path,
        tier_a_arxiv_path=tier_a_arxiv_path,
        tier_a_dois_path=tier_a_dois_path,
    )

    refs = _reference_records_from_seed_records(
        arxiv_records=arxiv_records,
        doi_records=doi_records,
        discovered_at_unix=discovered_at_unix,
        source_kind=source_kind,
    )
    if runs_dir is not None:
        run_artifact_path = _write_run_reference_artifact(
            runs_dir=runs_dir,
            run_id=effective_run_id,
            refs=refs,
        )
    if sqlite_path is not None:
        sqlite_store = SQLiteStore(sqlite_path)
        try:
            sqlite_store.create_schema()
            sqlite_store.upsert_reference_graph(
                refs=refs,
                resolved=[],
                run_id=effective_run_id,
                candidate_retry_days=candidate_retry_days,
                candidate_max_retries=candidate_max_retries,
            )
        finally:
            sqlite_store.close()

    return AwesomeSeedStats(
        repos_processed=len(repo_sources),
        total_matches=total_matches,
        unique_ids=unique_arxiv_count,
        repo_counts=dict(repo_counts),
        jsonl_path=jsonl_path,
        tier_a_seed_path=tier_a_seed_written_path,
        total_doi_matches=total_doi_matches,
        unique_dois=unique_doi_count,
        doi_repo_counts=dict(doi_repo_counts),
        doi_jsonl_path=doi_jsonl_path,
        tier_a_arxiv_path=tier_a_arxiv_written_path,
        tier_a_dois_path=tier_a_dois_path,
        run_id=effective_run_id,
        run_artifact_path=run_artifact_path,
    )


def discover_awesome_references(
    *,
    sources_path: Path,
    run_id: str,
    user_agent: str,
    runs_dir: Path,
    sqlite_path: Path | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 5,
    backoff_start_seconds: float = 1.0,
    backoff_cap_seconds: float = 30.0,
    delay_seconds: float = 0.2,
) -> list[ReferenceRecord]:
    out_dir = runs_dir / run_id
    stats = seed_awesome_sources(
        sources_path=sources_path,
        out_dir=out_dir,
        user_agent=user_agent,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_start_seconds=backoff_start_seconds,
        backoff_cap_seconds=backoff_cap_seconds,
        delay_seconds=delay_seconds,
        tier_a_seed_path=out_dir / "tierA_seed.txt",
        tier_a_arxiv_path=out_dir / "tierA_arxiv.txt",
        tier_a_dois_path=out_dir / "tierA_dois.txt",
        run_id=run_id,
        runs_dir=runs_dir,
        sqlite_path=sqlite_path,
    )
    if stats.run_artifact_path is None or not stats.run_artifact_path.exists():
        return []
    refs: list[ReferenceRecord] = []
    for line in stats.run_artifact_path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        refs.append(
            ReferenceRecord(
                ref_type=str(payload["ref_type"]),
                normalized_value=str(payload["normalized_value"]),
                source_kind=str(payload["source_kind"]),
                source_ref=str(payload["source_ref"]),
                discovered_at_unix=int(payload["discovered_at_unix"]),
            )
        )
    return refs


class SeedAwesomeService:
    def run(
        self,
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
        tier_a_arxiv_path: Path = DEFAULT_TIER_A_ARXIV_PATH,
        tier_a_dois_path: Path = DEFAULT_TIER_A_DOIS_PATH,
        run_id: str | None = None,
        runs_dir: Path | None = None,
        sqlite_path: Path | None = None,
        source_kind: str = "curated_repo",
        candidate_retry_days: int = 14,
        candidate_max_retries: int = 5,
    ) -> AwesomeSeedStats:
        return seed_awesome_sources(
            sources_path=sources_path,
            out_dir=out_dir,
            user_agent=user_agent,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            delay_seconds=delay_seconds,
            tier_a_seed_path=tier_a_seed_path,
            tier_a_arxiv_path=tier_a_arxiv_path,
            tier_a_dois_path=tier_a_dois_path,
            run_id=run_id,
            runs_dir=runs_dir,
            sqlite_path=sqlite_path,
            source_kind=source_kind,
            candidate_retry_days=candidate_retry_days,
            candidate_max_retries=candidate_max_retries,
        )
