from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from cv_rag.curation.s2_client import DEFAULT_FIELDS, MAX_BATCH_SIZE, SemanticScholarClient
from cv_rag.storage.sqlite import SQLiteStore

ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z-]+)?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)
DOI_ID_RE = re.compile(r"^(?:doi:)?10\.\d{4,9}/\S+$", re.IGNORECASE)


@dataclass(slots=True)
class CurateThresholds:
    tier0_min_citations: int = 200
    tier0_min_cpy: float = 30.0
    tier1_min_citations: int = 20
    tier1_min_cpy: float = 3.0


@dataclass(slots=True)
class CurateOptions:
    refresh_days: int = 30
    limit: int | None = None
    skip_non_arxiv: bool = True
    thresholds: CurateThresholds = field(default_factory=CurateThresholds)


@dataclass(slots=True)
class CurateResult:
    total_ids: int
    to_refresh: int
    updated: int
    skipped: int
    skipped_non_curatable: int
    tier_distribution: dict[int, int]


def is_curatable_paper_id(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if DOI_ID_RE.fullmatch(candidate) is not None:
        return True
    if candidate.upper().startswith("ARXIV:"):
        candidate = candidate.split(":", 1)[1].strip()
    return ARXIV_ID_RE.fullmatch(candidate) is not None


def load_venue_whitelist(path: Path) -> set[str]:
    if not path.exists():
        return set()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if raw is None:
            return set()
        if isinstance(raw, dict) and "venues" in raw:
            raw = raw["venues"]
        if not isinstance(raw, list):
            raise ValueError(f"Venue file must contain a list: {path}")
        return {str(item).strip() for item in raw if str(item).strip()}

    lines = path.read_text(encoding="utf-8").splitlines()
    out: set[str] = set()
    for line in lines:
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        out.add(cleaned)
    return out


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def compute_tier_and_score(
    *,
    citation_count: int,
    year: int | None,
    venue: str,
    publication_types: list[str],
    venue_whitelist: set[str],
    thresholds: CurateThresholds,
    current_year: int,
) -> tuple[int, float, int]:
    safe_citations = max(citation_count, 0)
    pub_year = year if year is not None else current_year
    citations_per_year = safe_citations / max(1, (current_year - pub_year + 1))
    venue_bonus = 1.0 if venue in venue_whitelist else 0.0
    score = math.log1p(safe_citations) + 0.5 * math.log1p(citations_per_year) + venue_bonus

    is_peer_reviewed = int(
        ("JournalArticle" in publication_types) or ("Conference" in publication_types)
    )
    if venue in venue_whitelist and (
        safe_citations >= thresholds.tier0_min_citations or citations_per_year >= thresholds.tier0_min_cpy
    ):
        tier = 0
    elif is_peer_reviewed and (
        safe_citations >= thresholds.tier1_min_citations or citations_per_year >= thresholds.tier1_min_cpy
    ):
        tier = 1
    else:
        tier = 2
    return tier, score, is_peer_reviewed


def _select_stale_or_missing_ids(
    arxiv_ids: list[str],
    timestamps: dict[str, int],
    refresh_days: int,
    now_ts: int,
) -> list[str]:
    if refresh_days < 0:
        refresh_days = 0
    cutoff = now_ts - (refresh_days * 86400)
    return [arxiv_id for arxiv_id in arxiv_ids if arxiv_id not in timestamps or timestamps[arxiv_id] < cutoff]


def _build_metric_row(
    *,
    arxiv_id: str,
    paper: dict[str, object],
    venue_whitelist: set[str],
    thresholds: CurateThresholds,
    now_ts: int,
    current_year: int,
) -> dict[str, object]:
    citation_count = _to_int(paper.get("citationCount")) or 0
    year = _to_int(paper.get("year"))
    venue = str(paper.get("venue") or "").strip()

    raw_publication_types = paper.get("publicationTypes")
    if isinstance(raw_publication_types, list):
        publication_types = [str(item) for item in raw_publication_types if str(item)]
    else:
        publication_types = []

    tier, score, is_peer_reviewed = compute_tier_and_score(
        citation_count=citation_count,
        year=year,
        venue=venue,
        publication_types=publication_types,
        venue_whitelist=venue_whitelist,
        thresholds=thresholds,
        current_year=current_year,
    )
    return {
        "arxiv_id": arxiv_id,
        "citation_count": citation_count,
        "year": year,
        "venue": venue,
        "publication_types": json.dumps(publication_types),
        "is_peer_reviewed": is_peer_reviewed,
        "score": score,
        "tier": tier,
        "updated_at": now_ts,
    }


def curate_corpus(
    *,
    sqlite_store: SQLiteStore,
    s2_client: SemanticScholarClient,
    venue_whitelist: set[str],
    options: CurateOptions,
    progress_callback: Callable[[int, int, str, str], None] | None = None,
) -> CurateResult:
    all_ids = sqlite_store.list_paper_arxiv_ids(limit=options.limit)
    skipped_non_curatable = 0
    if options.skip_non_arxiv:
        arxiv_ids = [item for item in all_ids if is_curatable_paper_id(item)]
        skipped_non_curatable = len(all_ids) - len(arxiv_ids)
    else:
        arxiv_ids = all_ids

    now_ts = int(datetime.now(UTC).timestamp())
    current_year = datetime.now(UTC).year

    if not arxiv_ids:
        return CurateResult(
            total_ids=len(all_ids),
            to_refresh=0,
            updated=0,
            skipped=len(all_ids),
            skipped_non_curatable=skipped_non_curatable,
            tier_distribution=sqlite_store.get_paper_metrics_tier_distribution(),
        )

    timestamps = sqlite_store.get_paper_metric_timestamps(arxiv_ids)
    to_refresh = _select_stale_or_missing_ids(arxiv_ids, timestamps, options.refresh_days, now_ts)
    skipped = (len(arxiv_ids) - len(to_refresh)) + skipped_non_curatable
    updated = 0

    total = len(to_refresh)
    processed = 0
    for offset in range(0, len(to_refresh), MAX_BATCH_SIZE):
        batch_ids = to_refresh[offset : offset + MAX_BATCH_SIZE]
        batch_responses = s2_client.get_papers_batch(batch_ids, fields=DEFAULT_FIELDS)
        rows: list[dict[str, object]] = []

        for idx, arxiv_id in enumerate(batch_ids):
            paper = batch_responses[idx] if idx < len(batch_responses) else None
            if paper is None:
                fallback = s2_client.get_paper(arxiv_id, fields=DEFAULT_FIELDS)
                if fallback is None:
                    skipped += 1
                    processed += 1
                    if progress_callback is not None:
                        progress_callback(processed, total, arxiv_id, "missing")
                    continue
                paper = fallback

            rows.append(
                _build_metric_row(
                    arxiv_id=arxiv_id,
                    paper=paper,
                    venue_whitelist=venue_whitelist,
                    thresholds=options.thresholds,
                    now_ts=now_ts,
                    current_year=current_year,
                )
            )
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total, arxiv_id, "updated")

        sqlite_store.upsert_paper_metrics(rows)
        updated += len(rows)

    return CurateResult(
        total_ids=len(all_ids),
        to_refresh=len(to_refresh),
        updated=updated,
        skipped=skipped,
        skipped_non_curatable=skipped_non_curatable,
        tier_distribution=sqlite_store.get_paper_metrics_tier_distribution(),
    )


class CurationService:
    def run(
        self,
        *,
        sqlite_store: SQLiteStore,
        s2_client: SemanticScholarClient,
        venue_whitelist: set[str],
        options: CurateOptions,
        progress_callback: Callable[[int, int, str, str], None] | None = None,
    ) -> CurateResult:
        return curate_corpus(
            sqlite_store=sqlite_store,
            s2_client=s2_client,
            venue_whitelist=venue_whitelist,
            options=options,
            progress_callback=progress_callback,
        )
