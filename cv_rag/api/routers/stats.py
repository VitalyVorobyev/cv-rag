from __future__ import annotations

from fastapi import APIRouter, Depends

from cv_rag.api.deps import get_app_settings, get_sqlite_store
from cv_rag.api.schemas import StatsResponse
from cv_rag.config import Settings
from cv_rag.sqlite_store import SQLiteStore

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
def get_stats(
    top_venues: int = 10,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
    settings: Settings = Depends(get_app_settings),
) -> StatsResponse:
    conn = sqlite_store.conn

    def scalar(query: str, params: tuple[object, ...] = ()) -> int:
        row = conn.execute(query, params).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    papers_count = scalar("SELECT COUNT(*) FROM papers")
    chunks_count = scalar("SELECT COUNT(*) FROM chunks")
    chunk_docs_count = scalar("SELECT COUNT(DISTINCT arxiv_id) FROM chunks")
    metrics_count = scalar("SELECT COUNT(*) FROM paper_metrics")
    papers_without_metrics = scalar(
        """
        SELECT COUNT(*)
        FROM papers p
        LEFT JOIN paper_metrics m ON p.arxiv_id = m.arxiv_id
        WHERE m.arxiv_id IS NULL
        """
    )

    tier_rows = conn.execute(
        "SELECT tier, COUNT(*) AS count FROM paper_metrics GROUP BY tier ORDER BY tier"
    ).fetchall()
    tier_distribution = {str(int(row["tier"])): int(row["count"]) for row in tier_rows if row["tier"] is not None}

    venue_rows: list[dict[str, object]] = []
    if top_venues > 0:
        raw = conn.execute(
            """
            SELECT venue, COUNT(*) AS count
            FROM paper_metrics
            WHERE venue IS NOT NULL AND TRIM(venue) != ''
            GROUP BY venue
            ORDER BY count DESC
            LIMIT ?
            """,
            (top_venues,),
        ).fetchall()
        venue_rows = [{"venue": str(row["venue"]), "count": int(row["count"])} for row in raw]

    pdf_files = len(list(settings.pdf_dir.glob("*.pdf")))
    tei_files = len(list(settings.tei_dir.glob("*.tei.xml")))

    return StatsResponse(
        papers_count=papers_count,
        chunks_count=chunks_count,
        chunk_docs_count=chunk_docs_count,
        metrics_count=metrics_count,
        papers_without_metrics=papers_without_metrics,
        pdf_files=pdf_files,
        tei_files=tei_files,
        tier_distribution=tier_distribution,
        top_venues=venue_rows,
    )
