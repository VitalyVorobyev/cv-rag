from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException

from cv_rag.interfaces.api.deps import get_sqlite_store
from cv_rag.interfaces.api.schemas import ChunkResponse, PaperDetailResponse, PaperListResponse, PaperSummary
from cv_rag.storage.sqlite import SQLiteStore

router = APIRouter(tags=["papers"])


def _parse_authors(authors_json: str | None) -> list[str]:
    if not authors_json:
        return []
    try:
        parsed = json.loads(authors_json)
        if isinstance(parsed, list):
            return [str(a) for a in parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [a.strip() for a in authors_json.split(",") if a.strip()]


def _public_arxiv_id(arxiv_id: str | None, doc_id: str | None) -> str | None:
    value = (arxiv_id or "").strip()
    if value and not value.startswith("doi:") and not value.startswith("url:"):
        return value
    if doc_id and doc_id.startswith("axv:"):
        raw = doc_id.removeprefix("axv:")
        return raw.split("v", 1)[0]
    return None


def _safe_row_value(row: object, key: str) -> object | None:
    try:
        return row[key]  # type: ignore[index]
    except Exception:  # noqa: BLE001
        return None


@router.get("/papers", response_model=PaperListResponse)
def list_papers(
    offset: int = 0,
    limit: int = 20,
    search: str = "",
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
) -> PaperListResponse:
    conn = sqlite_store.conn

    where_clause = ""
    params: list[object] = []
    if search.strip():
        where_clause = "WHERE p.arxiv_id LIKE ? OR p.doc_id LIKE ? OR p.title LIKE ?"
        like_pattern = f"%{search.strip()}%"
        params = [like_pattern, like_pattern, like_pattern]

    total_row = conn.execute(
        f"SELECT COUNT(*) FROM papers p {where_clause}",
        tuple(params),
    ).fetchone()
    total = int(total_row[0]) if total_row else 0

    rows = conn.execute(
        f"""
        SELECT
            p.arxiv_id,
            p.doc_id,
            p.title,
            p.summary,
            p.published,
            p.updated,
            p.authors_json,
            p.pdf_url,
            p.abs_url,
            COALESCE(cc.chunk_count, 0) AS chunk_count,
            m.tier,
            m.citation_count,
            m.venue
        FROM papers p
        LEFT JOIN (
            SELECT arxiv_id, COUNT(*) AS chunk_count
            FROM chunks
            GROUP BY arxiv_id
        ) cc ON p.arxiv_id = cc.arxiv_id
        LEFT JOIN paper_metrics m ON p.arxiv_id = m.arxiv_id
        {where_clause}
        ORDER BY p.published DESC, p.arxiv_id DESC
        LIMIT ? OFFSET ?
        """,
        (*params, limit, offset),
    ).fetchall()

    papers = [
        PaperSummary(
            doc_id=(
                str(_safe_row_value(row, "doc_id"))
                if _safe_row_value(row, "doc_id") is not None
                else None
            ),
            arxiv_id=_public_arxiv_id(
                str(_safe_row_value(row, "arxiv_id"))
                if _safe_row_value(row, "arxiv_id") is not None
                else None,
                str(_safe_row_value(row, "doc_id"))
                if _safe_row_value(row, "doc_id") is not None
                else None,
            ),
            title=str(row["title"]),
            summary=row["summary"],
            published=row["published"],
            updated=row["updated"],
            authors=_parse_authors(row["authors_json"]),
            pdf_url=row["pdf_url"],
            abs_url=row["abs_url"],
            chunk_count=int(row["chunk_count"]),
            tier=int(row["tier"]) if row["tier"] is not None else None,
            citation_count=int(row["citation_count"]) if row["citation_count"] is not None else None,
            venue=row["venue"],
        )
        for row in rows
    ]

    return PaperListResponse(papers=papers, total=total, offset=offset, limit=limit)


@router.get("/papers/{arxiv_id}", response_model=PaperDetailResponse)
def get_paper(
    arxiv_id: str,
    sqlite_store: SQLiteStore = Depends(get_sqlite_store),
) -> PaperDetailResponse:
    conn = sqlite_store.conn

    row = conn.execute(
        """
        SELECT
            p.arxiv_id,
            p.doc_id,
            p.title,
            p.summary,
            p.published,
            p.updated,
            p.authors_json,
            p.pdf_url,
            p.abs_url,
            m.tier,
            m.citation_count,
            m.venue
        FROM papers p
        LEFT JOIN paper_metrics m ON p.arxiv_id = m.arxiv_id
        WHERE p.arxiv_id = ?
        """,
        (arxiv_id,),
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Paper {arxiv_id} not found")

    chunk_rows = conn.execute(
        """
        SELECT chunk_id, doc_id, arxiv_id, title, section_title, chunk_index, text
        FROM chunks
        WHERE arxiv_id = ?
        ORDER BY chunk_index ASC
        """,
        (arxiv_id,),
    ).fetchall()

    paper = PaperSummary(
        doc_id=(
            str(_safe_row_value(row, "doc_id"))
            if _safe_row_value(row, "doc_id") is not None
            else None
        ),
        arxiv_id=_public_arxiv_id(
            str(_safe_row_value(row, "arxiv_id"))
            if _safe_row_value(row, "arxiv_id") is not None
            else None,
            str(_safe_row_value(row, "doc_id"))
            if _safe_row_value(row, "doc_id") is not None
            else None,
        ),
        title=str(row["title"]),
        summary=row["summary"],
        published=row["published"],
        updated=row["updated"],
        authors=_parse_authors(row["authors_json"]),
        pdf_url=row["pdf_url"],
        abs_url=row["abs_url"],
        chunk_count=len(chunk_rows),
        tier=int(row["tier"]) if row["tier"] is not None else None,
        citation_count=int(row["citation_count"]) if row["citation_count"] is not None else None,
        venue=row["venue"],
    )

    chunks = [
        ChunkResponse(
            chunk_id=str(cr["chunk_id"]),
            doc_id=(
                str(_safe_row_value(cr, "doc_id"))
                if _safe_row_value(cr, "doc_id") is not None
                else None
            ),
            arxiv_id=_public_arxiv_id(
                str(_safe_row_value(cr, "arxiv_id"))
                if _safe_row_value(cr, "arxiv_id") is not None
                else None,
                str(_safe_row_value(cr, "doc_id"))
                if _safe_row_value(cr, "doc_id") is not None
                else None,
            ),
            title=str(cr["title"]),
            section_title=str(cr["section_title"] or ""),
            text=str(cr["text"]),
            fused_score=0.0,
        )
        for cr in chunk_rows
    ]

    return PaperDetailResponse(paper=paper, chunks=chunks)
