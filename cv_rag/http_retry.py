from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def http_request_with_retry(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_start_seconds: float = 2.0,
    backoff_cap_seconds: float = 30.0,
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    error_label: str = "HTTP request",
    prepare_kwargs: Callable[[], dict[str, Any]] | None = None,
    **request_kwargs: Any,
) -> httpx.Response:
    """HTTP request with exponential backoff retry.

    Args:
        prepare_kwargs: Optional callable returning request kwargs, called on
            each attempt. Useful when kwargs must be rebuilt per attempt (e.g.
            reopening a file handle). When provided, ``request_kwargs`` is ignored.
    """
    attempts = max(1, max_retries)
    delay = max(backoff_start_seconds, 0.0)

    for attempt in range(1, attempts + 1):
        kwargs = prepare_kwargs() if prepare_kwargs is not None else request_kwargs

        try:
            response = client.request(method, url, **kwargs)
        except httpx.HTTPError as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"{error_label} failed after {attempts} attempts ({exc.__class__.__name__})"
                ) from exc
            time.sleep(min(delay, backoff_cap_seconds))
            delay = min(max(delay * 2, 1.0), backoff_cap_seconds)
            continue

        if response.status_code in retry_on_status:
            if attempt >= attempts:
                response.raise_for_status()
            retry_after = _parse_retry_after(response.headers.get("Retry-After"))
            sleep_seconds = retry_after if retry_after is not None else min(delay, backoff_cap_seconds)
            time.sleep(max(sleep_seconds, 0.0))
            delay = min(max(delay * 2, 1.0), backoff_cap_seconds)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError(f"{error_label} failed before receiving a response body")
