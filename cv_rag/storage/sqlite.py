from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from cv_rag.storage.repositories import (
    IngestCandidate,
    PaperRecord,
    ReferenceRecord,
    ResolvedReference,
    build_axv_doc_id,
    classify_provenance_kind,
    compute_candidate_priority,
)

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)


class SQLiteStore:
    def __init__(self, db_path: Path, *, check_same_thread: bool = True) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def _table_columns(self, table: str) -> set[str]:
        rows = self.conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {str(row["name"]) for row in rows}

    def _has_column(self, table: str, column: str) -> bool:
        return column in self._table_columns(table)

    def _ensure_column(self, table: str, column: str, column_def: str) -> None:
        if self._has_column(table, column):
            return
        self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    def create_schema(self) -> None:
        self.conn.executescript(
            """
            PRAGMA journal_mode=WAL;

            CREATE TABLE IF NOT EXISTS papers (
                arxiv_id TEXT PRIMARY KEY,
                doc_id TEXT UNIQUE,
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
                tei_path TEXT,
                provenance_kind TEXT,
                content_sha256 TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT,
                arxiv_id TEXT NOT NULL,
                title TEXT NOT NULL,
                section_title TEXT,
                chunk_index INTEGER,
                text TEXT NOT NULL,
                FOREIGN KEY (arxiv_id) REFERENCES papers(arxiv_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_arxiv_id ON chunks(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
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

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                arxiv_id TEXT,
                arxiv_id_with_version TEXT,
                doi TEXT,
                pdf_url TEXT,
                status TEXT NOT NULL,
                resolution_confidence REAL,
                provenance_kind TEXT,
                best_source_kind TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_documents_arxiv_id ON documents(arxiv_id);
            CREATE INDEX IF NOT EXISTS idx_documents_arxiv_version ON documents(arxiv_id_with_version);
            CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
            CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

            CREATE TABLE IF NOT EXISTS document_sources (
                doc_id TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                first_seen INTEGER NOT NULL,
                last_seen INTEGER NOT NULL,
                PRIMARY KEY (doc_id, source_kind, source_ref),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );

            CREATE TABLE IF NOT EXISTS reference_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ref_type TEXT NOT NULL,
                normalized_value TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                discovered_at_unix INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reference_events_run_id ON reference_events(run_id);

            CREATE TABLE IF NOT EXISTS resolution_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                arxiv_id TEXT,
                arxiv_id_with_version TEXT,
                doi TEXT,
                pdf_url TEXT,
                resolution_confidence REAL NOT NULL,
                resolved_at_unix INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_resolution_events_run_id ON resolution_events(run_id);

            CREATE TABLE IF NOT EXISTS ingest_candidates (
                doc_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                best_pdf_url TEXT,
                priority_score REAL NOT NULL,
                retry_count INTEGER NOT NULL DEFAULT 0,
                next_retry_unix INTEGER,
                last_error TEXT,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );
            CREATE INDEX IF NOT EXISTS idx_ingest_candidates_ready
                ON ingest_candidates(status, priority_score DESC, updated_at ASC);
            """
        )

        # Best-effort compatibility with older local DB files.
        self._ensure_column("papers", "doc_id", "doc_id TEXT")
        self._ensure_column("papers", "provenance_kind", "provenance_kind TEXT")
        self._ensure_column("papers", "content_sha256", "content_sha256 TEXT")
        self._ensure_column("chunks", "doc_id", "doc_id TEXT")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doc_id ON papers(doc_id)")
        self.conn.commit()

    def upsert_paper(
        self,
        paper: PaperRecord,
        pdf_path: Path | None = None,
        tei_path: Path | None = None,
    ) -> None:
        resolved_pdf_path = pdf_path if pdf_path is not None else getattr(paper, "pdf_path", None)
        resolved_tei_path = tei_path if tei_path is not None else getattr(paper, "tei_path", None)
        doc_id_value = getattr(paper, "doc_id", None)
        provenance_kind = getattr(paper, "provenance_kind", None)
        content_sha256 = getattr(paper, "content_sha256", None)
        doc_id = (doc_id_value or build_axv_doc_id(paper.arxiv_id_with_version)).strip()

        self.conn.execute(
            """
            INSERT INTO papers (
                arxiv_id,
                doc_id,
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
                tei_path,
                provenance_kind,
                content_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                doc_id=excluded.doc_id,
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
                tei_path=excluded.tei_path,
                provenance_kind=excluded.provenance_kind,
                content_sha256=COALESCE(excluded.content_sha256, papers.content_sha256)
            """,
            (
                paper.arxiv_id,
                doc_id,
                paper.arxiv_id_with_version,
                paper.version,
                paper.title,
                paper.summary,
                paper.published,
                paper.updated,
                ", ".join(paper.authors),
                paper.pdf_url,
                paper.abs_url,
                str(resolved_pdf_path) if resolved_pdf_path else None,
                str(resolved_tei_path) if resolved_tei_path else None,
                provenance_kind,
                content_sha256,
            ),
        )
        self.conn.commit()

    def upsert_chunks(self, chunk_rows: list[dict[str, Any]]) -> None:
        if not chunk_rows:
            return

        normalized_rows: list[dict[str, Any]] = []
        for row in chunk_rows:
            doc_id = row.get("doc_id")
            arxiv_id = str(row.get("arxiv_id", ""))
            if not doc_id and arxiv_id:
                doc_id = build_axv_doc_id(arxiv_id)
            normalized_row = dict(row)
            normalized_row["doc_id"] = doc_id
            normalized_rows.append(normalized_row)

        self.conn.executemany(
            """
            INSERT INTO chunks (chunk_id, doc_id, arxiv_id, title, section_title, chunk_index, text)
            VALUES (:chunk_id, :doc_id, :arxiv_id, :title, :section_title, :chunk_index, :text)
            ON CONFLICT(chunk_id) DO UPDATE SET
                doc_id=excluded.doc_id,
                arxiv_id=excluded.arxiv_id,
                title=excluded.title,
                section_title=excluded.section_title,
                chunk_index=excluded.chunk_index,
                text=excluded.text
            """,
            normalized_rows,
        )

        self.conn.executemany(
            "DELETE FROM chunks_fts WHERE chunk_id = ?",
            [(row["chunk_id"],) for row in normalized_rows],
        )

        if self._has_column("chunks_fts", "doc_id"):
            self.conn.executemany(
                """
                INSERT INTO chunks_fts (chunk_id, doc_id, arxiv_id, title, section_title, text)
                VALUES (:chunk_id, :doc_id, :arxiv_id, :title, :section_title, :text)
                """,
                normalized_rows,
            )
        else:
            self.conn.executemany(
                """
                INSERT INTO chunks_fts (chunk_id, arxiv_id, title, section_title, text)
                VALUES (:chunk_id, :arxiv_id, :title, :section_title, :text)
                """,
                normalized_rows,
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

    def list_legacy_unversioned_arxiv_ids(self) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT arxiv_id, arxiv_id_with_version
            FROM papers
            """
        ).fetchall()
        legacy_ids: set[str] = set()
        for row in rows:
            base_id = str(row["arxiv_id"]).strip()
            if not base_id:
                continue
            if ":" in base_id:
                continue
            raw_versioned = row["arxiv_id_with_version"]
            versioned = str(raw_versioned).strip() if raw_versioned is not None else ""
            if not versioned or not VERSION_SUFFIX_RE.search(versioned):
                legacy_ids.add(base_id)
        return sorted(legacy_ids)

    def update_paper_version_fields(
        self,
        *,
        arxiv_id: str,
        arxiv_id_with_version: str,
        version: str | None,
    ) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE papers
            SET arxiv_id_with_version = ?, version = ?, doc_id = ?
            WHERE arxiv_id = ?
            """,
            (arxiv_id_with_version, version, build_axv_doc_id(arxiv_id_with_version), arxiv_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

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

        fts_query = " OR ".join(f"{term}*" for term in terms)

        rows = self.conn.execute(
            """
            SELECT
                c.chunk_id,
                c.doc_id,
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
                "doc_id": row["doc_id"],
                "arxiv_id": row["arxiv_id"],
                "title": row["title"],
                "section_title": row["section_title"],
                "text": row["text"],
                "score": float(row["bm25_score"]),
            }
            for row in rows
        ]

    def upsert_reference_graph(
        self,
        *,
        refs: list[ReferenceRecord],
        resolved: list[ResolvedReference],
        run_id: str,
        candidate_retry_days: int,
        candidate_max_retries: int,
        now_unix: int | None = None,
    ) -> None:
        now = int(time.time()) if now_unix is None else now_unix

        for ref in refs:
            doc_id = ref.doc_id
            provenance_kind = classify_provenance_kind(ref.source_kind)

            arxiv_id = None
            arxiv_id_with_version = None
            doi = None
            pdf_url = None
            resolution_kind = "metadata_only"
            status = "discovered"
            resolution_confidence = 0.0
            if ref.ref_type == "arxiv":
                arxiv_id_with_version = ref.normalized_value
                arxiv_id = ref.normalized_value.split("v", 1)[0]
                resolution_kind = (
                    "arxiv_versioned"
                    if VERSION_SUFFIX_RE.search(ref.normalized_value)
                    else "metadata_only"
                )
                if VERSION_SUFFIX_RE.search(ref.normalized_value):
                    status = "ready"
                    resolution_confidence = 0.9
                    pdf_url = f"https://arxiv.org/pdf/{ref.normalized_value}.pdf"
            elif ref.ref_type == "doi":
                doi = ref.normalized_value
            elif ref.ref_type == "pdf_url":
                pdf_url = ref.normalized_value
                resolution_kind = "pdf_only"
                status = "ready"
                resolution_confidence = 0.6

            self.conn.execute(
                """
                INSERT INTO documents (
                    doc_id, arxiv_id, arxiv_id_with_version, doi, pdf_url, status,
                    resolution_confidence, provenance_kind, best_source_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    arxiv_id=COALESCE(excluded.arxiv_id, documents.arxiv_id),
                    arxiv_id_with_version=COALESCE(excluded.arxiv_id_with_version, documents.arxiv_id_with_version),
                    doi=COALESCE(excluded.doi, documents.doi),
                    pdf_url=COALESCE(excluded.pdf_url, documents.pdf_url),
                    status=CASE
                        WHEN documents.status = 'ingested' THEN documents.status
                        WHEN documents.status = 'ready' AND excluded.status = 'discovered' THEN documents.status
                        ELSE excluded.status
                    END,
                    resolution_confidence=MAX(documents.resolution_confidence, excluded.resolution_confidence),
                    provenance_kind=COALESCE(excluded.provenance_kind, documents.provenance_kind),
                    best_source_kind=excluded.best_source_kind,
                    updated_at=excluded.updated_at
                """,
                (
                    doc_id,
                    arxiv_id,
                    arxiv_id_with_version,
                    doi,
                    pdf_url,
                    status,
                    resolution_confidence,
                    provenance_kind,
                    ref.source_kind,
                    ref.discovered_at_unix,
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO document_sources (doc_id, source_kind, source_ref, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id, source_kind, source_ref) DO UPDATE SET
                    last_seen=excluded.last_seen
                """,
                (
                    doc_id,
                    ref.source_kind,
                    ref.source_ref,
                    ref.discovered_at_unix,
                    now,
                ),
            )
            self.conn.execute(
                """
                INSERT INTO reference_events (
                    run_id, ref_type, normalized_value, source_kind, source_ref, discovered_at_unix
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    ref.ref_type,
                    ref.normalized_value,
                    ref.source_kind,
                    ref.source_ref,
                    ref.discovered_at_unix,
                ),
            )

            priority = compute_candidate_priority(
                source_kind=ref.source_kind,
                resolution_confidence=resolution_confidence,
                age_days=0,
                retry_count=0,
                resolution_kind=resolution_kind,
            )
            self.conn.execute(
                """
                INSERT INTO ingest_candidates (
                    doc_id, status, best_pdf_url, priority_score, retry_count, next_retry_unix, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status=CASE
                        WHEN ingest_candidates.status = 'ingested' THEN ingest_candidates.status
                        WHEN ingest_candidates.status = 'ready'
                             AND excluded.status IN ('discovered', 'resolved')
                        THEN ingest_candidates.status
                        ELSE excluded.status
                    END,
                    priority_score=MAX(ingest_candidates.priority_score, excluded.priority_score),
                    best_pdf_url=COALESCE(ingest_candidates.best_pdf_url, excluded.best_pdf_url),
                    next_retry_unix=COALESCE(ingest_candidates.next_retry_unix, excluded.next_retry_unix),
                    updated_at=excluded.updated_at
                """,
                (doc_id, status, pdf_url, priority, 0, None, None, now),
            )

        for item in resolved:
            resolved_at = item.resolved_at_unix if item.resolved_at_unix is not None else now
            source_kind = item.source_kind or "openalex_resolved"
            provenance_kind = classify_provenance_kind(source_kind)
            resolution_kind = "metadata_only"
            if item.arxiv_id_with_version:
                resolution_kind = "arxiv_versioned"
            elif item.pdf_url and item.doi:
                resolution_kind = "oa_pdf"
            elif item.pdf_url:
                resolution_kind = "pdf_only"

            status = "ready" if item.arxiv_id_with_version or item.pdf_url else "blocked"
            next_retry_unix = None
            if status == "blocked":
                next_retry_unix = now + (max(candidate_retry_days, 1) * 86400)

            # When DOI resolves to an arXiv canonical doc, keep the DOI identity as a resolved alias
            # and remove it from the ready queue to prevent duplicate ingestion.
            if item.doi and item.doc_id.startswith("axv:"):
                alias_doc_id = f"doi:{item.doi}"
                if alias_doc_id != item.doc_id:
                    self.conn.execute(
                        """
                        INSERT INTO documents (
                            doc_id, arxiv_id, arxiv_id_with_version, doi, pdf_url, status,
                            resolution_confidence, provenance_kind, best_source_kind, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(doc_id) DO UPDATE SET
                            arxiv_id=COALESCE(excluded.arxiv_id, documents.arxiv_id),
                            arxiv_id_with_version=COALESCE(
                                excluded.arxiv_id_with_version,
                                documents.arxiv_id_with_version
                            ),
                            doi=COALESCE(excluded.doi, documents.doi),
                            pdf_url=COALESCE(excluded.pdf_url, documents.pdf_url),
                            status='resolved',
                            resolution_confidence=MAX(documents.resolution_confidence, excluded.resolution_confidence),
                            best_source_kind=excluded.best_source_kind,
                            updated_at=excluded.updated_at
                        """,
                        (
                            alias_doc_id,
                            item.arxiv_id,
                            item.arxiv_id_with_version,
                            item.doi,
                            item.pdf_url,
                            "resolved",
                            item.resolution_confidence,
                            provenance_kind,
                            source_kind,
                            resolved_at,
                            now,
                        ),
                    )
                    self.conn.execute(
                        """
                        INSERT INTO ingest_candidates (
                            doc_id, status, best_pdf_url, priority_score, retry_count,
                            next_retry_unix, last_error, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(doc_id) DO UPDATE SET
                            status=CASE
                                WHEN ingest_candidates.status = 'ingested' THEN ingest_candidates.status
                                ELSE 'resolved'
                            END,
                            best_pdf_url=COALESCE(excluded.best_pdf_url, ingest_candidates.best_pdf_url),
                            next_retry_unix=NULL,
                            last_error=excluded.last_error,
                            updated_at=excluded.updated_at
                        """,
                        (
                            alias_doc_id,
                            "resolved",
                            item.pdf_url,
                            0.0,
                            0,
                            None,
                            f"merged_to:{item.doc_id}",
                            now,
                        ),
                    )

            self.conn.execute(
                """
                INSERT INTO resolution_events (
                    run_id,
                    doc_id,
                    arxiv_id,
                    arxiv_id_with_version,
                    doi,
                    pdf_url,
                    resolution_confidence,
                    resolved_at_unix
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    item.doc_id,
                    item.arxiv_id,
                    item.arxiv_id_with_version,
                    item.doi,
                    item.pdf_url,
                    item.resolution_confidence,
                    resolved_at,
                ),
            )

            self.conn.execute(
                """
                INSERT INTO documents (
                    doc_id, arxiv_id, arxiv_id_with_version, doi, pdf_url, status,
                    resolution_confidence, provenance_kind, best_source_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    arxiv_id=COALESCE(excluded.arxiv_id, documents.arxiv_id),
                    arxiv_id_with_version=COALESCE(excluded.arxiv_id_with_version, documents.arxiv_id_with_version),
                    doi=COALESCE(excluded.doi, documents.doi),
                    pdf_url=COALESCE(excluded.pdf_url, documents.pdf_url),
                    status=CASE
                        WHEN documents.status = 'ingested' THEN documents.status
                        ELSE excluded.status
                    END,
                    resolution_confidence=MAX(documents.resolution_confidence, excluded.resolution_confidence),
                    provenance_kind=COALESCE(excluded.provenance_kind, documents.provenance_kind),
                    best_source_kind=excluded.best_source_kind,
                    updated_at=excluded.updated_at
                """,
                (
                    item.doc_id,
                    item.arxiv_id,
                    item.arxiv_id_with_version,
                    item.doi,
                    item.pdf_url,
                    status,
                    item.resolution_confidence,
                    provenance_kind,
                    source_kind,
                    resolved_at,
                    now,
                ),
            )
            priority = compute_candidate_priority(
                source_kind=source_kind,
                resolution_confidence=item.resolution_confidence,
                age_days=0,
                retry_count=0,
                resolution_kind=resolution_kind,
            )
            self.conn.execute(
                """
                INSERT INTO ingest_candidates (
                    doc_id, status, best_pdf_url, priority_score, retry_count, next_retry_unix, last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    status=CASE
                        WHEN ingest_candidates.status = 'ingested' THEN ingest_candidates.status
                        ELSE excluded.status
                    END,
                    best_pdf_url=COALESCE(excluded.best_pdf_url, ingest_candidates.best_pdf_url),
                    priority_score=excluded.priority_score,
                    next_retry_unix=excluded.next_retry_unix,
                    updated_at=excluded.updated_at
                """,
                (
                    item.doc_id,
                    status,
                    item.pdf_url,
                    priority,
                    0,
                    next_retry_unix,
                    None,
                    now,
                ),
            )

        self.conn.commit()

    def list_ready_candidates(self, *, limit: int, now_unix: int | None = None) -> list[IngestCandidate]:
        now = int(time.time()) if now_unix is None else now_unix
        rows = self.conn.execute(
            """
            SELECT doc_id, status, best_pdf_url, priority_score, retry_count, next_retry_unix
            FROM ingest_candidates
            WHERE status = 'ready'
              AND (next_retry_unix IS NULL OR next_retry_unix <= ?)
            ORDER BY priority_score DESC, updated_at ASC
            LIMIT ?
            """,
            (now, limit),
        ).fetchall()
        return [
            IngestCandidate(
                doc_id=str(row["doc_id"]),
                status=str(row["status"]),
                best_pdf_url=str(row["best_pdf_url"]) if row["best_pdf_url"] else None,
                priority_score=float(row["priority_score"]),
                retry_count=int(row["retry_count"]),
                next_retry_unix=int(row["next_retry_unix"]) if row["next_retry_unix"] is not None else None,
            )
            for row in rows
        ]

    def get_documents_by_ids(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        placeholders = ", ".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"""
            SELECT doc_id, arxiv_id, arxiv_id_with_version, doi, pdf_url, status, provenance_kind
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            tuple(doc_ids),
        ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                "arxiv_id": row["arxiv_id"],
                "arxiv_id_with_version": row["arxiv_id_with_version"],
                "doi": row["doi"],
                "pdf_url": row["pdf_url"],
                "status": row["status"],
                "provenance_kind": row["provenance_kind"],
            }
            for row in rows
        ]

    def mark_candidate_result(
        self,
        *,
        doc_id: str,
        status: str,
        reason: str | None,
        candidate_retry_days: int,
        candidate_max_retries: int,
        now_unix: int | None = None,
    ) -> None:
        now = int(time.time()) if now_unix is None else now_unix
        row = self.conn.execute(
            """
            SELECT retry_count
            FROM ingest_candidates
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()
        retry_count = int(row["retry_count"]) if row is not None else 0
        next_retry_unix: int | None = None
        final_status = status

        if status in {"blocked", "failed"}:
            retry_count += 1
            if retry_count >= max(candidate_max_retries, 1):
                final_status = "failed"
                next_retry_unix = None
            else:
                final_status = "blocked"
                next_retry_unix = now + (max(candidate_retry_days, 1) * 86400)

        self.conn.execute(
            """
            INSERT INTO ingest_candidates (
                doc_id, status, best_pdf_url, priority_score, retry_count, next_retry_unix, last_error, updated_at
            ) VALUES (?, ?, NULL, 0.0, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                status=excluded.status,
                retry_count=excluded.retry_count,
                next_retry_unix=excluded.next_retry_unix,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at
            """,
            (doc_id, final_status, retry_count, next_retry_unix, reason, now),
        )
        self.conn.execute(
            """
            UPDATE documents
            SET status = ?, updated_at = ?
            WHERE doc_id = ?
            """,
            (final_status, now, doc_id),
        )
        self.conn.commit()

    def get_doc_provenance_kinds(self, doc_ids: list[str]) -> dict[str, str]:
        if not doc_ids:
            return {}
        placeholders = ", ".join("?" for _ in doc_ids)
        rows = self.conn.execute(
            f"""
            SELECT doc_id, provenance_kind
            FROM documents
            WHERE doc_id IN ({placeholders})
            """,
            tuple(doc_ids),
        ).fetchall()
        out: dict[str, str] = {}
        for row in rows:
            kind = row["provenance_kind"]
            if kind is None:
                continue
            out[str(row["doc_id"])] = str(kind)
        return out
