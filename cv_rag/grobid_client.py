from __future__ import annotations

import time
from pathlib import Path

import httpx


def pdf_to_tei(
    pdf_path: Path,
    grobid_url: str,
    timeout_seconds: float = 180.0,
    max_retries: int = 8,
    backoff_start_seconds: float = 2.0,
    backoff_cap_seconds: float = 20.0,
) -> str:
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    delay = backoff_start_seconds

    for attempt in range(1, max_retries + 1):
        with pdf_path.open("rb") as handle:
            files = {"input": (pdf_path.name, handle, "application/pdf")}
            try:
                response = httpx.post(endpoint, files=files, timeout=timeout_seconds)
            except httpx.HTTPError as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"GROBID request failed after {max_retries} attempts for {pdf_path}"
                    ) from exc
                time.sleep(min(delay, backoff_cap_seconds))
                delay = min(delay * 2, backoff_cap_seconds)
                continue

        if response.status_code == 503:
            if attempt == max_retries:
                raise RuntimeError(
                    f"GROBID returned 503 after {max_retries} attempts for {pdf_path}"
                )
            time.sleep(min(delay, backoff_cap_seconds))
            delay = min(delay * 2, backoff_cap_seconds)
            continue

        if response.is_error:
            raise RuntimeError(
                f"GROBID request failed for {pdf_path} with status {response.status_code}: "
                f"{response.text[:500]}"
            )

        return response.text

    raise RuntimeError(f"Unable to parse {pdf_path} with GROBID")
