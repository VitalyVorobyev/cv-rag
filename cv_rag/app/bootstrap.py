from __future__ import annotations

from dataclasses import dataclass

from cv_rag.embeddings import OllamaEmbedClient
from cv_rag.retrieval.hybrid import HybridRetriever
from cv_rag.shared.settings import Settings, get_settings
from cv_rag.storage.qdrant import QdrantStore
from cv_rag.storage.sqlite import SQLiteStore


@dataclass(slots=True)
class AppRuntime:
    settings: Settings
    sqlite_store: SQLiteStore
    qdrant_store: QdrantStore
    retriever: HybridRetriever


def build_retriever_runtime(*, settings: Settings | None = None, check_same_thread: bool = True) -> AppRuntime:
    use_settings = settings or get_settings()
    use_settings.ensure_directories()

    sqlite_store = SQLiteStore(use_settings.sqlite_path, check_same_thread=check_same_thread)
    sqlite_store.create_schema()
    qdrant_store = QdrantStore(
        url=use_settings.qdrant_url,
        collection_name=use_settings.qdrant_collection,
    )
    embed_client = OllamaEmbedClient(
        base_url=use_settings.ollama_url,
        model=use_settings.ollama_model,
        timeout_seconds=use_settings.http_timeout_seconds,
    )
    retriever = HybridRetriever(
        embedder=embed_client,
        qdrant_store=qdrant_store,
        sqlite_store=sqlite_store,
    )
    return AppRuntime(
        settings=use_settings,
        sqlite_store=sqlite_store,
        qdrant_store=qdrant_store,
        retriever=retriever,
    )
