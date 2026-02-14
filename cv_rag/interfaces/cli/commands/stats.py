from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cv_rag.shared.settings import Settings
from cv_rag.storage.sqlite import SQLiteStore


def run_stats_command(
    *,
    settings: Settings,
    console: Console,
    top_venues: int,
    sqlite_store_cls: type[SQLiteStore] = SQLiteStore,
) -> None:
    settings.ensure_directories()

    if not settings.sqlite_path.exists():
        console.print(f"[red]SQLite database not found: {settings.sqlite_path}[/red]")
        raise typer.Exit(code=1)

    sqlite_store = sqlite_store_cls(settings.sqlite_path)
    sqlite_store.create_schema()

    def scalar(query: str, params: tuple[object, ...] = ()) -> int:
        row = sqlite_store.conn.execute(query, params).fetchone()
        if row is None:
            return 0
        return int(row[0] or 0)

    try:
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
        indexed_pdf_rows = sqlite_store.conn.execute(
            """
            SELECT pdf_path
            FROM papers
            WHERE pdf_path IS NOT NULL
              AND TRIM(pdf_path) != ''
            """
        ).fetchall()
        indexed_pdf_names = {Path(str(row[0])).name for row in indexed_pdf_rows if row[0]}

        tier_rows = sqlite_store.conn.execute(
            """
            SELECT tier, COUNT(*) AS count
            FROM paper_metrics
            GROUP BY tier
            ORDER BY tier
            """
        ).fetchall()
        venue_rows = []
        if top_venues > 0:
            venue_rows = sqlite_store.conn.execute(
                """
                SELECT venue, COUNT(*) AS count
                FROM paper_metrics
                WHERE venue IS NOT NULL
                  AND TRIM(venue) != ''
                GROUP BY venue
                ORDER BY count DESC
                LIMIT ?
                """,
                (top_venues,),
            ).fetchall()
    finally:
        sqlite_store.close()

    pdf_files = list(settings.pdf_dir.glob("*.pdf"))
    tei_files = list(settings.tei_dir.glob("*.tei.xml"))
    pdf_file_names = {path.name for path in pdf_files}
    orphan_pdf_files = pdf_file_names - indexed_pdf_names

    table = Table(title="Database Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("sqlite path", str(settings.sqlite_path))
    table.add_row("papers rows", str(papers_count))
    table.add_row("chunks rows", str(chunks_count))
    table.add_row("chunk docs (distinct arxiv_id)", str(chunk_docs_count))
    table.add_row("paper_metrics rows", str(metrics_count))
    table.add_row("papers without metrics", str(papers_without_metrics))
    table.add_row("pdf files on disk", str(len(pdf_files)))
    table.add_row("tei files on disk", str(len(tei_files)))
    table.add_row("indexed pdf paths in papers", str(len(indexed_pdf_names)))
    table.add_row("orphan pdf files (disk - indexed)", str(len(orphan_pdf_files)))
    console.print(table)

    if tier_rows:
        tier_table = Table(title="Tier Distribution")
        tier_table.add_column("Tier", justify="right")
        tier_table.add_column("Count", justify="right")
        for row in tier_rows:
            tier_table.add_row(str(int(row["tier"])), str(int(row["count"])))
        console.print(tier_table)

    if venue_rows:
        venue_table = Table(title=f"Top Venues (top {top_venues})")
        venue_table.add_column("Venue")
        venue_table.add_column("Count", justify="right")
        for row in venue_rows:
            venue_table.add_row(str(row["venue"]), str(int(row["count"])))
        console.print(venue_table)
