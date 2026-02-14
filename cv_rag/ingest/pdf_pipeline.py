from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.ingest.arxiv_client import PaperMetadata, download_pdf, write_metadata_json
from cv_rag.ingest.chunking import chunk_sections
from cv_rag.ingest.grobid_client import pdf_to_tei
from cv_rag.ingest.tei_extract import extract_sections
from cv_rag.retrieval.hybrid import format_citation
from cv_rag.shared.settings import Settings, get_settings
from cv_rag.storage.qdrant import QdrantStore, VectorPoint
from cv_rag.storage.repositories import PaperRecord
from cv_rag.storage.sqlite import SQLiteStore

logger = logging.getLogger(__name__)


def load_tei_or_parse(pdf_path: Path, tei_path: Path, settings: Settings, force: bool) -> str:
    if tei_path.exists() and not force:
        return tei_path.read_text(encoding="utf-8")

    tei_xml = pdf_to_tei(
        pdf_path=pdf_path,
        grobid_url=settings.grobid_url,
        timeout_seconds=settings.http_timeout_seconds,
        max_retries=settings.grobid_max_retries,
        backoff_start_seconds=settings.grobid_backoff_start_seconds,
        backoff_cap_seconds=settings.grobid_backoff_cap_seconds,
    )
    tei_path.write_text(tei_xml, encoding="utf-8")
    return tei_xml


@dataclass
class IngestResult:
    papers_processed: int = 0
    total_chunks: int = 0
    failed_papers: list[str] = field(default_factory=list)


class IngestPipeline:
    """Orchestrates paper ingestion: download, parse, chunk, embed, store."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def run(
        self,
        papers: list[PaperMetadata],
        metadata_json_path: Path,
        force_grobid: bool = False,
        embed_batch_size: int | None = None,
        on_progress: Callable[[int, int, PaperMetadata], None] | None = None,
    ) -> IngestResult:
        """Run the full ingest pipeline.

        Args:
            on_progress: Optional callback(index, total, paper) for progress reporting.
        """
        settings = self.settings
        use_batch_size = embed_batch_size or settings.embed_batch_size
        result = IngestResult()

        if not papers:
            return result

        write_metadata_json(papers, metadata_json_path)

        sqlite_store = SQLiteStore(settings.sqlite_path)
        sqlite_store.create_schema()
        qdrant_store = QdrantStore(
            url=settings.qdrant_url,
            collection_name=settings.qdrant_collection,
        )
        embed_client = OllamaEmbedClient(
            base_url=settings.ollama_url,
            model=settings.ollama_model,
            timeout_seconds=settings.http_timeout_seconds,
        )

        collection_initialized = False

        try:
            for idx, paper in enumerate(papers, start=1):
                if on_progress:
                    on_progress(idx, len(papers), paper)
                try:
                    pdf_path = download_pdf(
                        paper=paper,
                        pdf_dir=settings.pdf_dir,
                        timeout_seconds=settings.http_timeout_seconds,
                        user_agent=settings.user_agent,
                    )
                    tei_path = settings.tei_dir / f"{paper.safe_file_stem()}.tei.xml"
                    tei_xml = load_tei_or_parse(
                        pdf_path=pdf_path, tei_path=tei_path, settings=settings, force=force_grobid,
                    )

                    sections = extract_sections(tei_xml)
                    chunks = chunk_sections(
                        sections,
                        max_chars=settings.chunk_max_chars,
                        overlap_chars=settings.chunk_overlap_chars,
                    )
                    if not chunks:
                        continue

                    chunk_texts = [c.text for c in chunks]
                    vectors = embed_client.embed_in_batches(chunk_texts, batch_size=use_batch_size)
                    if len(vectors) != len(chunks):
                        raise RuntimeError(
                            f"Embedding count mismatch for {paper.arxiv_id}: "
                            f"{len(vectors)} vectors for {len(chunks)} chunks"
                        )

                    if vectors and not collection_initialized:
                        qdrant_store.ensure_collection(len(vectors[0]))
                        collection_initialized = True

                    sqlite_store.upsert_paper(
                        PaperRecord(
                            arxiv_id=paper.arxiv_id,
                            arxiv_id_with_version=paper.arxiv_id_with_version,
                            version=paper.version,
                            title=paper.title,
                            summary=paper.summary,
                            published=paper.published,
                            updated=paper.updated,
                            authors=paper.authors,
                            pdf_url=paper.pdf_url,
                            abs_url=paper.abs_url,
                            pdf_path=pdf_path,
                            tei_path=tei_path,
                        )
                    )

                    sqlite_rows: list[dict[str, object]] = []
                    points: list[VectorPoint] = []
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        chunk_id = f"{paper.arxiv_id}:{chunk.chunk_index}"
                        sqlite_rows.append({
                            "chunk_id": chunk_id,
                            "arxiv_id": paper.arxiv_id,
                            "title": paper.title,
                            "section_title": chunk.section_title,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                        })
                        points.append(VectorPoint(
                            point_id=chunk_id,
                            vector=vector,
                            payload={
                                "chunk_id": chunk_id,
                                "arxiv_id": paper.arxiv_id,
                                "title": paper.title,
                                "section_title": chunk.section_title,
                                "text": chunk.text,
                                "citation": format_citation(paper.arxiv_id, chunk.section_title),
                            },
                        ))

                    sqlite_store.upsert_chunks(sqlite_rows)
                    qdrant_store.upsert(points)
                    result.total_chunks += len(chunks)
                    result.papers_processed += 1
                except Exception as exc:  # noqa: BLE001
                    result.failed_papers.append(f"{paper.arxiv_id}: {exc}")
        finally:
            sqlite_store.close()

        return result
