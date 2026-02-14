from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from cv_rag.app.bootstrap import build_retriever_runtime
from cv_rag.interfaces.api.routers import answer, health, papers, search, stats

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    runtime = build_retriever_runtime(check_same_thread=False)

    app.state.settings = runtime.settings
    app.state.sqlite_store = runtime.sqlite_store
    app.state.retriever = runtime.retriever

    logger.info("cv-rag API started (sqlite=%s)", runtime.settings.sqlite_path)
    yield

    runtime.sqlite_store.close()
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
