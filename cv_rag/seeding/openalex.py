from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from cv_rag.ingest.arxiv_client import normalize_arxiv_id
from cv_rag.seeding.doi import normalize_doi
from cv_rag.shared.http import http_request_with_retry

logger = logging.getLogger(__name__)

OPENALEX_WORKS_BY_DOI_URL = "https://api.openalex.org/works/doi:"
DEFAULT_TIER_A_OPENALEX_URLS_PATH = Path("data/curation/tierA_urls_openalex.txt")


@dataclass(slots=True, frozen=True)
class OpenAlexOAFields:
    pdf_url: str | None
    landing_page_url: str | None
    license: str | None
    is_oa: bool | None
    source: str


@dataclass(slots=True, frozen=True)
class OpenAlexResolvedRecord:
    doi: str
    openalex_id: str | None
    arxiv_id: str | None
    is_oa: bool | None
    pdf_url: str | None
    landing_page_url: str | None
    license: str | None
    source: str
    fetched_at: str

    def to_dict(self) -> dict[str, object | None]:
        return {
            "doi": self.doi,
            "openalex_id": self.openalex_id,
            "arxiv_id": self.arxiv_id,
            "is_oa": self.is_oa,
            "pdf_url": self.pdf_url,
            "landing_page_url": self.landing_page_url,
            "license": self.license,
            "source": self.source,
            "fetched_at": self.fetched_at,
        }


@dataclass(slots=True)
class ResolveStats:
    dois_processed: int
    resolved_records: int
    resolved_pdf_urls: int
    resolved_arxiv_ids: int
    unresolved: int
    cache_hits: int
    jsonl_path: Path
    tier_a_urls_path: Path
    tier_a_arxiv_path: Path | None
    cache_dir: Path


def _normalize_optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalize_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _location_pdf_url(location: object) -> str | None:
    if not isinstance(location, dict):
        return None
    return _normalize_optional_text(location.get("pdf_url"))


def _location_landing_url(location: object) -> str | None:
    if not isinstance(location, dict):
        return None
    return _normalize_optional_text(location.get("landing_page_url"))


def _location_license(location: object) -> str | None:
    if not isinstance(location, dict):
        return None
    return _normalize_optional_text(location.get("license"))


def _location_is_oa(location: object) -> bool | None:
    if not isinstance(location, dict):
        return None
    return _normalize_optional_bool(location.get("is_oa"))


def extract_openalex_oa_fields(work: dict[str, Any]) -> OpenAlexOAFields:
    best = work.get("best_oa_location")
    primary = work.get("primary_location")
    locations = work.get("locations")

    best_pdf = _location_pdf_url(best)
    if best_pdf:
        return OpenAlexOAFields(
            pdf_url=best_pdf,
            landing_page_url=_location_landing_url(best),
            license=_location_license(best),
            is_oa=_location_is_oa(best),
            source="best_oa_location",
        )

    primary_pdf = _location_pdf_url(primary)
    if primary_pdf:
        return OpenAlexOAFields(
            pdf_url=primary_pdf,
            landing_page_url=_location_landing_url(primary),
            license=_location_license(primary),
            is_oa=_location_is_oa(primary),
            source="primary_location",
        )

    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            if not bool(location.get("is_oa")):
                continue
            location_pdf = _location_pdf_url(location)
            if not location_pdf:
                continue
            return OpenAlexOAFields(
                pdf_url=location_pdf,
                landing_page_url=_location_landing_url(location),
                license=_location_license(location),
                is_oa=_location_is_oa(location),
                source="locations",
            )

    fallback_location = best if isinstance(best, dict) else primary if isinstance(primary, dict) else None
    return OpenAlexOAFields(
        pdf_url=None,
        landing_page_url=_location_landing_url(fallback_location),
        license=_location_license(fallback_location),
        is_oa=_location_is_oa(fallback_location),
        source="none",
    )


def extract_openalex_arxiv_id(work: dict[str, Any]) -> str | None:
    ids = work.get("ids")
    if not isinstance(ids, dict):
        return None
    raw_arxiv = _normalize_optional_text(ids.get("arxiv"))
    if not raw_arxiv:
        return None
    normalized = normalize_arxiv_id(raw_arxiv)
    return normalized or None


def load_dois(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"DOI file not found: {path}")

    dois: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        normalized = normalize_doi(stripped)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        dois.append(normalized)
    return dois


def _cache_file_for_doi(cache_dir: Path, doi: str) -> Path:
    key = hashlib.sha256(doi.encode("utf-8")).hexdigest()
    return cache_dir / f"{key}.json"


def _write_cache(cache_file: Path, payload: dict[str, Any]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _read_cache(cache_file: Path) -> dict[str, Any] | None:
    if not cache_file.exists():
        return None
    try:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring invalid cache entry %s (%s)", cache_file, exc)
        return None
    return cached if isinstance(cached, dict) else None


def _fetch_work_json(
    *,
    client: httpx.Client,
    doi: str,
    api_key: str | None,
    max_retries: int,
    backoff_start_seconds: float,
    backoff_cap_seconds: float,
    cache_dir: Path,
) -> tuple[dict[str, Any] | None, bool]:
    cache_file = _cache_file_for_doi(cache_dir, doi)
    cached = _read_cache(cache_file)
    if cached is not None:
        if cached.get("_status") == 404:
            return None, True
        return cached, True

    url = f"{OPENALEX_WORKS_BY_DOI_URL}{quote(doi, safe='')}"
    headers: dict[str, str] = {}
    params: dict[str, str] = {}
    if api_key:
        headers["X-Api-Key"] = api_key
        params["api_key"] = api_key

    try:
        response = http_request_with_retry(
            client,
            "GET",
            url,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label=f"OpenAlex request ({doi})",
            headers=headers if headers else None,
            params=params if params else None,
        )
    except RuntimeError as exc:
        logger.warning("%s", exc)
        return None, False
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {401, 403}:
            raise PermissionError(
                "OpenAlex API key required; set OPENALEX_API_KEY "
                "or pass --api-key-env with a configured variable."
            ) from exc
        if status_code == 404:
            _write_cache(cache_file, {"_status": 404, "doi": doi})
            return None, False
        logger.warning("OpenAlex request failed (%s) for DOI %s", status_code, doi)
        return None, False

    try:
        payload = response.json()
    except ValueError:
        logger.warning("OpenAlex returned non-JSON payload for DOI %s", doi)
        return None, False
    if not isinstance(payload, dict):
        logger.warning("OpenAlex payload is not an object for DOI %s", doi)
        return None, False

    _write_cache(cache_file, payload)
    return payload, False


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        if lines:
            file_handle.write("\n".join(lines))
            file_handle.write("\n")


def resolve_dois_openalex(
    *,
    dois_path: Path,
    out_dir: Path,
    user_agent: str,
    email: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 5,
    backoff_start_seconds: float = 1.0,
    backoff_cap_seconds: float = 30.0,
    delay_seconds: float = 0.2,
    cache_path: Path | None = None,
    tier_a_urls_path: Path = DEFAULT_TIER_A_OPENALEX_URLS_PATH,
    tier_a_arxiv_path: Path | None = None,
) -> ResolveStats:
    dois = load_dois(dois_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "openalex_resolved.jsonl"
    cache_dir = cache_path or (out_dir / "openalex_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    user_agent_header = user_agent.strip()
    if email and email.strip():
        user_agent_header = f"{user_agent_header} (mailto:{email.strip()})"
    headers = {"User-Agent": user_agent_header, "Accept": "application/json"}

    cache_hits = 0
    resolved_records = 0
    unresolved = 0
    resolved_pdf_by_doi: dict[str, str] = {}
    resolved_arxiv_by_doi: dict[str, str] = {}

    with (
        httpx.Client(timeout=timeout_seconds, headers=headers, follow_redirects=True) as client,
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
    ):
        for idx, doi in enumerate(dois):
            if idx > 0 and delay_seconds > 0:
                time.sleep(delay_seconds)

            try:
                work_json, cache_hit = _fetch_work_json(
                    client=client,
                    doi=doi,
                    api_key=api_key,
                    max_retries=max_retries,
                    backoff_start_seconds=backoff_start_seconds,
                    backoff_cap_seconds=backoff_cap_seconds,
                    cache_dir=cache_dir,
                )
            except PermissionError as exc:
                raise RuntimeError(str(exc)) from None
            if cache_hit:
                cache_hits += 1

            if work_json is None:
                unresolved += 1
                continue

            oa_fields = extract_openalex_oa_fields(work_json)
            arxiv_id = extract_openalex_arxiv_id(work_json)
            record = OpenAlexResolvedRecord(
                doi=doi,
                openalex_id=_normalize_optional_text(work_json.get("id")),
                arxiv_id=arxiv_id,
                is_oa=oa_fields.is_oa,
                pdf_url=oa_fields.pdf_url,
                landing_page_url=oa_fields.landing_page_url,
                license=oa_fields.license,
                source=oa_fields.source,
                fetched_at=datetime.now(UTC).isoformat(),
            )
            jsonl_file.write(json.dumps(record.to_dict(), ensure_ascii=False))
            jsonl_file.write("\n")
            resolved_records += 1

            if record.pdf_url:
                resolved_pdf_by_doi[doi] = record.pdf_url
            if record.arxiv_id:
                resolved_arxiv_by_doi[doi] = record.arxiv_id

    sorted_pdf_urls_by_doi = sorted(resolved_pdf_by_doi.items(), key=lambda item: item[0])
    unique_urls: list[str] = []
    seen_urls: set[str] = set()
    for _, pdf_url in sorted_pdf_urls_by_doi:
        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)
        unique_urls.append(pdf_url)

    unique_arxiv: list[str] = []
    if tier_a_arxiv_path is not None:
        unique_arxiv = sorted(set(resolved_arxiv_by_doi.values()))

    _write_lines(tier_a_urls_path, unique_urls)
    if tier_a_arxiv_path is not None:
        _write_lines(tier_a_arxiv_path, unique_arxiv)

    return ResolveStats(
        dois_processed=len(dois),
        resolved_records=resolved_records,
        resolved_pdf_urls=len(unique_urls),
        resolved_arxiv_ids=len(unique_arxiv),
        unresolved=unresolved,
        cache_hits=cache_hits,
        jsonl_path=jsonl_path,
        tier_a_urls_path=tier_a_urls_path,
        tier_a_arxiv_path=tier_a_arxiv_path,
        cache_dir=cache_dir,
    )


class ResolveDoisService:
    def run(
        self,
        *,
        dois_path: Path,
        out_dir: Path,
        user_agent: str,
        email: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 5,
        backoff_start_seconds: float = 1.0,
        backoff_cap_seconds: float = 30.0,
        delay_seconds: float = 0.2,
        cache_path: Path | None = None,
        tier_a_urls_path: Path = DEFAULT_TIER_A_OPENALEX_URLS_PATH,
        tier_a_arxiv_path: Path | None = None,
    ) -> ResolveStats:
        return resolve_dois_openalex(
            dois_path=dois_path,
            out_dir=out_dir,
            user_agent=user_agent,
            email=email,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            delay_seconds=delay_seconds,
            cache_path=cache_path,
            tier_a_urls_path=tier_a_urls_path,
            tier_a_arxiv_path=tier_a_arxiv_path,
        )
