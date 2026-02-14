from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from cv_rag.arxiv_sync import PaperMetadata

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def create_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                arxiv_id_with_version TEXT,
                version TEXT,
                title TEXT NOT NULL,
                summary TEXT,
                published TEXT,
                updated TEXT,
                authors_json TEXT,
                pdf_url TEXT,
                abs_url TEXT,
                pdf_path TEXT,
                tei_path TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                arxiv_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section_title TEXT,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON chunks(arxiv_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                arxiv_id,
                title,
                section_title,
                text
            );

            CREATE TABLE IF NOT EXISTS paper_metrics (
                arxiv_id TEXT PRIMARY KEY,
                citation_count INTEGER,
                year INTEGER,
                venue TEXT,
                publication_types TEXT,
                is_peer_reviewed INTEGER,
                score REAL,
                tier INTEGER,
                updated_at INTEGER,
                FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
            );

            CREATE INDEX IF NOT EXISTS idx_paper_metrics_tier ON paper_metrics(tier);
            """
        )
        self.conn.commit()

    def upsert_paper(
        self,
        paper: PaperMetadata,
        pdf_path: Path | None,
        tei_path: Path | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO papers (
                arxiv_id,
                arxiv_id_with_version,
                version,
                title,
                summary,
                published,
                updated,
                authors_json,
                pdf_url,
                abs_url,
                pdf_path,
                tei_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                arxiv_id_with_version=excluded.arxiv_id_with_version,
                version=excluded.version,
                title=excluded.title,
                summary=excluded.summary,
                published=excluded.published,
                updated=excluded.updated,
                authors_json=excluded.authors_json,
                pdf_url=excluded.pdf_url,
                abs_url=excluded.abs_url,
                pdf_path=excluded.pdf_path,
                tei_path=excluded.tei_path
            """,
            (
                paper.arxiv_id,
                paper.arxiv_id_with_version,
                paper.version,
                paper.title,
                paper.summary,
                paper.published,
                paper.updated,
                ", ".join(paper.authors),
                paper.pdf_url,
                paper.abs_url,
                str(pdf_path) if pdf_path else None,
                str(tei_path) if tei_path else None,
            ),
        )
        self.conn.commit()

    def upsert_chunks(self, chunk_rows: list[dict[str, Any]]) -> None:
        if not chunk_rows:
            return

        self.conn.executemany(
            """
            INSERT INTO chunks (chunk_id, arxiv_id, title, section_title, chunk_index, text)
            VALUES (:chunk_id, :arxiv_id, :title, :section_title, :chunk_index, :text)
            ON CONFLICT(chunk_id) DO UPDATE SET
                arxiv_id=excluded.arxiv_id,
                title=excluded.title,
                section_title=excluded.section_title,
                chunk_index=excluded.chunk_index,
                text=excluded.text
            """,
            chunk_rows,
        )

        self.conn.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?",
            [(row["chunk_id"],) for row in chunk_rows],
        )
        self.conn.executemany(
            """
            INSERT INTO chunks_fts (chunk_id, arxiv_id, title, section_title, text)
            VALUES (:chunk_id, :arxiv_id, :title, :section_title, :text)
            """,
            chunk_rows,
        )
        self.conn.commit()

    def get_ingested_versioned_ids(self) -> set[str]:
        rows = self.conn.execute(
            """
            SELECT arxiv_id_with_version
            FROM papers
            WHERE arxiv_id_with_version IS NOT NULL
              AND TRIM(arxiv_id_with_version) != ''
            """
        ).fetchall()
        return {
            str(row["arxiv_id_with_version"]).strip()
            for row in rows
            if str(row["arxiv_id_with_version"]).strip()
        }

    def list_paper_arxiv_ids(self, limit: int | None = None) -> list[str]:
        sql = "SELECT arxiv_id FROM papers ORDER BY arxiv_id ASC"
        params: tuple[object, ...]
        if limit is None:
            params = ()
        else:
            sql = f"{sql} LIMIT ?"
            params = (limit,)
        rows = self.conn.execute(sql, params).fetchall()
        return [str(row["arxiv_id"]) for row in rows]

    def get_paper_metric_timestamps(self, arxiv_ids: list[str]) -> dict[str, int]:
        if not arxiv_ids:
            return {}
        placeholders = ", ".join("?" for _ in arxiv_ids)
        rows = self.conn.execute(
            f"""
            SELECT arxiv_id, updated_at
            FROM paper_metrics
            WHERE arxiv_id IN ({placeholders})
            """,
            tuple(arxiv_ids),
        ).fetchall()
        out: dict[str, int] = {}
        for row in rows:
            updated_at = row["updated_at"]
            if updated_at is None:
                continue
            out[str(row["arxiv_id"])] = int(updated_at)
        return out

    def upsert_paper_metrics(self, metric_rows: list[dict[str, Any]]) -> None:
        if not metric_rows:
            return
        self.conn.executemany(
            """
            INSERT INTO paper_metrics (
                arxiv_id,
                citation_count,
                year,
                venue,
                publication_types,
                is_peer_reviewed,
                score,
                tier,
                updated_at
            )
            VALUES (
                :arxiv_id,
                :citation_count,
                :year,
                :venue,
                :publication_types,
                :is_peer_reviewed,
                :score,
                :tier,
                :updated_at
            )
            ON CONFLICT(arxiv_id) DO UPDATE SET
                citation_count=excluded.citation_count,
                year=excluded.year,
                venue=excluded.venue,
                publication_types=excluded.publication_types,
                is_peer_reviewed=excluded.is_peer_reviewed,
                score=excluded.score,
                tier=excluded.tier,
                updated_at=excluded.updated_at
            """,
            metric_rows,
        )
        self.conn.commit()

    def get_paper_tiers(self, arxiv_ids: list[str]) -> dict[str, int]:
        if not arxiv_ids:
            return {}
        placeholders = ", ".join("?" for _ in arxiv_ids)
        rows = self.conn.execute(
            f"""
            SELECT arxiv_id, tier
            FROM paper_metrics
            WHERE arxiv_id IN ({placeholders})
            """,
            tuple(arxiv_ids),
        ).fetchall()
        out: dict[str, int] = {}
        for row in rows:
            tier = row["tier"]
            if tier is None:
                continue
            out[str(row["arxiv_id"])] = int(tier)
        return out

    def get_paper_metrics_tier_distribution(self) -> dict[int, int]:
        rows = self.conn.execute(
            """
            SELECT tier, COUNT(*) AS count
            FROM paper_metrics
            GROUP BY tier
            """
        ).fetchall()
        return {int(row["tier"]): int(row["count"]) for row in rows if row["tier"] is not None}

    def keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        terms = TOKEN_RE.findall(query)
        if not terms:
            return []

        # Prefix matching over cleaned tokens is a simple, robust FTS query.
        fts_query = " OR ".join(f"{term}*" for term in terms)

        rows = self.conn.execute(
            """
            SELECT
                c.chunk_id,
                c.arxiv_id,
                c.title,
                c.section_title,
                c.text,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()

        return [
            {
                "chunk_id": row["chunk_id"],
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "section_title": row["section_title"],
                "text": row["text"],
                "score": float(row["bm25_score"]),
            }
            for row in rows
        ]
