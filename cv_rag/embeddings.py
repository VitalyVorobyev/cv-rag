from __future__ import annotations

from typing import Iterable

import httpx


class OllamaEmbedClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        endpoint = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()

        if "embeddings" in data:
            embeddings = data["embeddings"]
            if len(embeddings) != len(texts):
                raise RuntimeError(
                    "Ollama returned a mismatched number of embeddings: "
                    f"expected {len(texts)}, got {len(embeddings)}"
                )
            return embeddings

        if "embedding" in data and len(texts) == 1:
            return [data["embedding"]]

        raise RuntimeError(f"Unexpected Ollama embed response shape: {data}")

    def embed_in_batches(self, texts: Iterable[str], batch_size: int = 16) -> list[list[float]]:
        source = list(texts)
        out: list[list[float]] = []
        for i in range(0, len(source), batch_size):
            out.extend(self.embed_texts(source[i : i + batch_size]))
        return out
