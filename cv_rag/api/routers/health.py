from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends

from cv_rag.api.deps import get_app_settings
from cv_rag.api.schemas import HealthResponse, ServiceHealth
from cv_rag.config import Settings
from cv_rag.qdrant_store import QdrantStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(settings: Settings = Depends(get_app_settings)) -> HealthResponse:
    services: list[ServiceHealth] = []

    # Qdrant
    try:
        root = httpx.get(settings.qdrant_url, timeout=5.0)
        root.raise_for_status()
        version = root.json().get("version", "unknown")
        qdrant_store = QdrantStore(settings.qdrant_url, settings.qdrant_collection)
        qdrant_store.client.get_collections()
        services.append(ServiceHealth(service="Qdrant", status="ok", detail=f"version={version}"))
    except Exception as exc:  # noqa: BLE001
        services.append(ServiceHealth(service="Qdrant", status="fail", detail=str(exc)))

    # GROBID
    try:
        alive = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/isalive", timeout=5.0)
        alive.raise_for_status()
        isalive_text = alive.text.strip()
        version_resp = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/version", timeout=5.0)
        grobid_version = version_resp.text.strip() if version_resp.status_code < 500 else "unknown"
        services.append(
            ServiceHealth(service="GROBID", status="ok", detail=f"alive={isalive_text}; version={grobid_version}")
        )
    except Exception as exc:  # noqa: BLE001
        services.append(ServiceHealth(service="GROBID", status="fail", detail=str(exc)))

    # Ollama
    try:
        version_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/version", timeout=5.0)
        version_resp.raise_for_status()
        version = version_resp.json().get("version", "unknown")
        tags_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        tags_resp.raise_for_status()
        model_count = len(tags_resp.json().get("models", []))
        services.append(
            ServiceHealth(service="Ollama", status="ok", detail=f"version={version}; models={model_count}")
        )
    except Exception as exc:  # noqa: BLE001
        services.append(ServiceHealth(service="Ollama", status="fail", detail=str(exc)))

    return HealthResponse(services=services)
