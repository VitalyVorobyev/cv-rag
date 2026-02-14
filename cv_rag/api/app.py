from __future__ import annotations

import logging
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cv_rag.api.routers import answer, health, papers, search, stats
from cv_rag.config import get_settings
from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.qdrant_store import QdrantStore
from cv_rag.retrieve import HybridRetriever
from cv_rag.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_directories()

    # Create SQLiteStore with check_same_thread=False so FastAPI's
    # threadpool workers can use the connection created in the main thread.
    # Safe because: WAL mode, read-heavy workload, writes only via CLI.
    sqlite_store = SQLiteStore.__new__(SQLiteStore)
    sqlite_store.db_path = settings.sqlite_path
    sqlite_store.db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite_store.conn = sqlite3.connect(str(settings.sqlite_path), check_same_thread=False)
    sqlite_store.conn.row_factory = sqlite3.Row
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
    retriever = HybridRetriever(
        embedder=embed_client,
        qdrant_store=qdrant_store,
        sqlite_store=sqlite_store,
    )

    app.state.settings = settings
    app.state.sqlite_store = sqlite_store
    app.state.retriever = retriever

    logger.info("cv-rag API started (sqlite=%s)", settings.sqlite_path)
    yield

    sqlite_store.close()
    logger.info("cv-rag API shut down")


def create_app() -> FastAPI:
    app = FastAPI(title="cv-rag", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")
    app.include_router(stats.router, prefix="/api")
    app.include_router(papers.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(answer.router, prefix="/api")

    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")

    return app
