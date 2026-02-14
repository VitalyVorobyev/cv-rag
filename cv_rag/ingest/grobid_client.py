from __future__ import annotations

from pathlib import Path

import httpx

from cv_rag.shared.http import http_request_with_retry


def pdf_to_tei(
    pdf_path: Path,
    grobid_url: str,
    timeout_seconds: float = 180.0,
    max_retries: int = 8,
    backoff_start_seconds: float = 2.0,
    backoff_cap_seconds: float = 20.0,
) -> str:
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"

    def _prepare_kwargs() -> dict:
        handle = pdf_path.open("rb")
        return {"files": {"input": (pdf_path.name, handle, "application/pdf")}}

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True) as client:
        response = http_request_with_retry(
            client, "POST", endpoint,
            max_retries=max_retries,
            backoff_start_seconds=backoff_start_seconds,
            backoff_cap_seconds=backoff_cap_seconds,
            error_label=f"GROBID request for {pdf_path}",
            prepare_kwargs=_prepare_kwargs,
        )
        return response.text
