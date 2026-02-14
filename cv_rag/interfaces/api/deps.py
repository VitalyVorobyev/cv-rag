from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.requests import Request

if TYPE_CHECKING:
    from cv_rag.retrieval.hybrid import HybridRetriever
    from cv_rag.shared.settings import Settings
    from cv_rag.storage.sqlite import SQLiteStore


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever  # type: ignore[no-any-return]


def get_sqlite_store(request: Request) -> SQLiteStore:
    return request.app.state.sqlite_store  # type: ignore[no-any-return]


def get_app_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]
