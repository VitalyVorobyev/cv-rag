from __future__ import annotations

import os
from urllib.parse import quote

import httpx

from cv_rag.shared.http import http_request_with_retry

S2_BASE_URL = "https://api.semanticscholar.org"
DEFAULT_FIELDS = ("title", "year", "citationCount", "venue", "publicationTypes", "url")
MAX_BATCH_SIZE = 500


def _dedupe_preserve(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _doi_value(value: str) -> str | None:
    candidate = value.strip()
    lowered = candidate.casefold()
    if lowered.startswith("doi:"):
        candidate = candidate.split(":", 1)[1].strip()
        lowered = candidate.casefold()
    if lowered.startswith("10."):
        return candidate
    return None


def _paper_id_candidates(arxiv_id: str) -> list[str]:
    value = arxiv_id.strip()
    if not value:
        return []
    if value.startswith("http://") or value.startswith("https://"):
        return [value]
    doi = _doi_value(value)
    if doi is not None:
        return _dedupe_preserve([f"DOI:{doi}", value, f"https://doi.org/{doi}"])
    candidates = [value]
    if not value.upper().startswith("ARXIV:"):
        candidates.append(f"ARXIV:{value}")
        candidates.append(f"https://arxiv.org/abs/{value}")
    return _dedupe_preserve(candidates)


def _batch_paper_id(arxiv_id: str) -> str:
    value = arxiv_id.strip()
    if not value:
        return value
    if value.startswith("http://") or value.startswith("https://"):
        return value
    doi = _doi_value(value)
    if doi is not None:
        return f"DOI:{doi}"
    if value.upper().startswith("ARXIV:"):
        return value
    return f"ARXIV:{value}"


class SemanticScholarClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = S2_BASE_URL,
        timeout_seconds: float = 30.0,
        user_agent: str = "cv-rag/0.1 (+local)",
        max_retries: int = 5,
        backoff_start_seconds: float = 2.0,
        backoff_cap_seconds: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent
        self.max_retries = max_retries
        self.backoff_start_seconds = backoff_start_seconds
        self.backoff_cap_seconds = backoff_cap_seconds

    def _headers(self) -> dict[str, str]:
        headers = {"User-Agent": self.user_agent}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _request(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        json: dict[str, object] | None = None,
        error_label: str,
    ) -> httpx.Response:
        with httpx.Client(
            timeout=self.timeout_seconds,
            headers=self._headers(),
            follow_redirects=True,
        ) as client:
            return http_request_with_retry(
                client,
                method,
                url,
                params=params,
                json=json,
                max_retries=self.max_retries,
                backoff_start_seconds=self.backoff_start_seconds,
                backoff_cap_seconds=self.backoff_cap_seconds,
                retry_on_status=(429, 500, 502, 503, 504),
                error_label=error_label,
            )

    def get_paper(
        self,
        arxiv_id: str,
        fields: tuple[str, ...] = DEFAULT_FIELDS,
    ) -> dict[str, object] | None:
        params = {"fields": ",".join(fields)}
        last_not_found: httpx.HTTPStatusError | None = None
        for paper_id in _paper_id_candidates(arxiv_id):
            encoded = quote(paper_id, safe="")
            url = f"{self.base_url}/graph/v1/paper/{encoded}"
            try:
                response = self._request(
                    method="GET",
                    url=url,
                    params=params,
                    error_label=f"S2 get_paper({paper_id})",
                )
            except httpx.HTTPStatusError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    last_not_found = exc
                    continue
                raise
            data = response.json()
            return data if isinstance(data, dict) else None

        if last_not_found is not None:
            return None
        return None

    def get_papers_batch(
        self,
        list_of_arxiv_ids: list[str],
        fields: tuple[str, ...] = DEFAULT_FIELDS,
    ) -> list[dict[str, object] | None]:
        if len(list_of_arxiv_ids) > MAX_BATCH_SIZE:
            raise ValueError(f"S2 batch accepts up to {MAX_BATCH_SIZE} ids, got {len(list_of_arxiv_ids)}")
        if not list_of_arxiv_ids:
            return []

        body = {"ids": [_batch_paper_id(arxiv_id) for arxiv_id in list_of_arxiv_ids]}
        response = self._request(
            method="POST",
            url=f"{self.base_url}/graph/v1/paper/batch",
            params={"fields": ",".join(fields)},
            json=body,
            error_label="S2 get_papers_batch",
        )
        data = response.json()
        if not isinstance(data, list):
            return []
        out: list[dict[str, object] | None] = []
        for item in data:
            if isinstance(item, dict):
                out.append(item)
            else:
                out.append(None)
        return out
