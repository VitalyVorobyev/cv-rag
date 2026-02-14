from __future__ import annotations

import uuid

import httpx
from rich.console import Console
from rich.table import Table

from cv_rag.shared.settings import Settings
from cv_rag.storage.qdrant import QdrantStore, VectorPoint


def run_doctor_command(
    *,
    settings: Settings,
    console: Console,
    qdrant_test_point: bool,
    qdrant_store_cls: type[QdrantStore] = QdrantStore,
) -> None:
    table = Table(title="Service Health")
    table.add_column("Service")
    table.add_column("Status")
    table.add_column("Version/Detail")

    # Qdrant checks
    try:
        root = httpx.get(settings.qdrant_url, timeout=5.0)
        root.raise_for_status()
        version = root.json().get("version", "unknown") if "json" in dir(root) else "unknown"

        qdrant_store = qdrant_store_cls(settings.qdrant_url, settings.qdrant_collection)
        qdrant_store.client.get_collections()
        table.add_row("Qdrant", "ok", f"version={version}")

        if qdrant_test_point:
            doctor_collection = f"{settings.qdrant_collection}_doctor"
            doctor_store = qdrant_store_cls(settings.qdrant_url, doctor_collection)
            doctor_store.ensure_collection(3)
            point_id = str(uuid.uuid4())
            doctor_store.upsert(
                [
                    VectorPoint(
                        point_id=point_id,
                        vector=[0.1, 0.2, 0.3],
                        payload={"kind": "doctor", "chunk_id": point_id},
                    )
                ]
            )
            table.add_row("Qdrant test insert", "ok", f"collection={doctor_collection}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Qdrant", "fail", str(exc))

    # GROBID checks
    try:
        alive = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/isalive", timeout=5.0)
        alive.raise_for_status()
        isalive_text = alive.text.strip()

        version_resp = httpx.get(f"{settings.grobid_url.rstrip('/')}/api/version", timeout=5.0)
        grobid_version = version_resp.text.strip() if version_resp.status_code < 500 else "unknown"
        table.add_row("GROBID", "ok", f"alive={isalive_text}; version={grobid_version}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("GROBID", "fail", str(exc))

    # Ollama checks
    try:
        version_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/version", timeout=5.0)
        version_resp.raise_for_status()
        version = version_resp.json().get("version", "unknown")

        tags_resp = httpx.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=5.0)
        tags_resp.raise_for_status()
        model_count = len(tags_resp.json().get("models", []))
        table.add_row("Ollama", "ok", f"version={version}; models={model_count}")
    except Exception as exc:  # noqa: BLE001
        table.add_row("Ollama", "fail", str(exc))

    console.print(table)
