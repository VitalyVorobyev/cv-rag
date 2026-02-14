from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class Settings(BaseModel):
    qdrant_url: str = "http://localhost:6333"
    grobid_url: str = "http://localhost:8070"
    ollama_url: str = "http://localhost:11434"
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    data_dir: Path = Path("data")
    pdf_dir: Path = Path("data/pdfs")
    tei_dir: Path = Path("data/tei")
    metadata_dir: Path = Path("data/metadata")
    metadata_json_path: Path = Path("data/metadata/arxiv_cs_cv.json")
    sqlite_path: Path = Path("data/cv_rag.sqlite3")
    qdrant_collection: str = "cv_papers"
    ollama_model: str = "nomic-embed-text"
    default_arxiv_limit: int = 50
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 200
    http_timeout_seconds: float = 120.0
    arxiv_max_retries: int = 5
    arxiv_backoff_start_seconds: float = 2.0
    arxiv_backoff_cap_seconds: float = 30.0
    grobid_max_retries: int = 8
    grobid_backoff_start_seconds: float = 2.0
    grobid_backoff_cap_seconds: float = 20.0
    embed_batch_size: int = 16
    relevance_vector_threshold: float = 0.45
    user_agent: str = Field(default="cv-rag/0.1 (+local)")

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.tei_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = Path(os.getenv("CV_RAG_DATA_DIR", "data"))
    pdf_dir = Path(os.getenv("CV_RAG_PDF_DIR", str(data_dir / "pdfs")))
    tei_dir = Path(os.getenv("CV_RAG_TEI_DIR", str(data_dir / "tei")))
    metadata_dir = Path(os.getenv("CV_RAG_METADATA_DIR", str(data_dir / "metadata")))

    settings = Settings(
        qdrant_url=os.getenv("CV_RAG_QDRANT_URL", "http://localhost:6333"),
        grobid_url=os.getenv("CV_RAG_GROBID_URL", "http://localhost:8070"),
        ollama_url=os.getenv("CV_RAG_OLLAMA_URL", "http://localhost:11434"),
        arxiv_api_url=os.getenv("CV_RAG_ARXIV_API_URL", "https://export.arxiv.org/api/query"),
        data_dir=data_dir,
        pdf_dir=pdf_dir,
        tei_dir=tei_dir,
        metadata_dir=metadata_dir,
        metadata_json_path=Path(
            os.getenv("CV_RAG_METADATA_JSON", str(metadata_dir / "arxiv_cs_cv.json"))
        ),
        sqlite_path=Path(os.getenv("CV_RAG_SQLITE_PATH", str(data_dir / "cv_rag.sqlite3"))),
        qdrant_collection=os.getenv("CV_RAG_QDRANT_COLLECTION", "cv_papers"),
        ollama_model=os.getenv("CV_RAG_OLLAMA_MODEL", "nomic-embed-text"),
        default_arxiv_limit=int(os.getenv("CV_RAG_DEFAULT_LIMIT", "50")),
        chunk_max_chars=int(os.getenv("CV_RAG_CHUNK_MAX_CHARS", "1200")),
        chunk_overlap_chars=int(os.getenv("CV_RAG_CHUNK_OVERLAP", "200")),
        http_timeout_seconds=float(os.getenv("CV_RAG_HTTP_TIMEOUT", "120")),
        arxiv_max_retries=int(os.getenv("CV_RAG_ARXIV_MAX_RETRIES", "5")),
        arxiv_backoff_start_seconds=float(os.getenv("CV_RAG_ARXIV_BACKOFF_START", "2")),
        arxiv_backoff_cap_seconds=float(os.getenv("CV_RAG_ARXIV_BACKOFF_CAP", "30")),
        grobid_max_retries=int(os.getenv("CV_RAG_GROBID_MAX_RETRIES", "8")),
        grobid_backoff_start_seconds=float(os.getenv("CV_RAG_GROBID_BACKOFF_START", "2")),
        grobid_backoff_cap_seconds=float(os.getenv("CV_RAG_GROBID_BACKOFF_CAP", "20")),
        embed_batch_size=int(os.getenv("CV_RAG_EMBED_BATCH_SIZE", "16")),
        user_agent=os.getenv("CV_RAG_USER_AGENT", "cv-rag/0.1 (+local)"),
    )
    return settings
