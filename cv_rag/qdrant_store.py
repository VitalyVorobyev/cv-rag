from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


@dataclass(slots=True)
class VectorPoint:
    point_id: str
    vector: list[float]
    payload: dict[str, Any]


class QdrantStore:
    def __init__(self, url: str, collection_name: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int) -> None:
        collection_names = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in collection_names:
            info = self.client.get_collection(self.collection_name)
            vectors = info.config.params.vectors
            existing_size: int | None = None
            if hasattr(vectors, "size"):
                existing_size = int(vectors.size)
            elif isinstance(vectors, dict):
                first = next(iter(vectors.values()), None)
                if first is not None and hasattr(first, "size"):
                    existing_size = int(first.size)

            if existing_size is not None and existing_size != vector_size:
                raise RuntimeError(
                    f"Collection {self.collection_name} already exists with vector size "
                    f"{existing_size}, but incoming vectors have size {vector_size}."
                )
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=vector_size,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def upsert(self, points: list[VectorPoint], batch_size: int = 64) -> None:
        if not points:
            return

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    qmodels.PointStruct(
                        id=self._normalize_point_id(p.point_id),
                        vector=p.vector,
                        payload=p.payload,
                    )
                    for p in batch
                ],
                wait=True,
            )

    @staticmethod
    def _normalize_point_id(point_id: str | int) -> str | int:
        if isinstance(point_id, int):
            return point_id
        text_id = str(point_id)
        try:
            uuid.UUID(text_id)
            return text_id
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, text_id))

    def search(self, query_vector: list[float], limit: int) -> list[dict[str, Any]]:
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        except AttributeError:
            query_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = list(getattr(query_result, "points", []))

        out: list[dict[str, Any]] = []
        for item in results:
            payload = dict(getattr(item, "payload", {}) or {})
            point_id = payload.get("chunk_id") or str(getattr(item, "id", ""))
            out.append(
                {
                    "chunk_id": point_id,
                    "arxiv_id": payload.get("arxiv_id", ""),
                    "title": payload.get("title", ""),
                    "section_title": payload.get("section_title", ""),
                    "text": payload.get("text", ""),
                    "score": float(getattr(item, "score", 0.0)),
                }
            )
        return out
