from __future__ import annotations

from typing import Iterable

import httpx


class OllamaEmbedClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._endpoint_mode: str | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        with httpx.Client(timeout=self.timeout_seconds) as client:
            try:
                return self._embed_texts_with_client(client, texts, allow_model_fallback=True)
            except httpx.HTTPStatusError as exc:
                detail = self._extract_error_message(exc.response) if exc.response is not None else str(exc)
                raise RuntimeError(f"Ollama embedding request failed: {detail}") from exc

    def _embed_texts_with_client(
        self,
        client: httpx.Client,
        texts: list[str],
        allow_model_fallback: bool,
    ) -> list[list[float]]:
        try:
            if self._endpoint_mode != "legacy":
                try:
                    return self._embed_batch_endpoint(client, texts)
                except httpx.HTTPStatusError as exc:
                    if exc.response is None or exc.response.status_code != 404:
                        raise
                    self._endpoint_mode = "legacy"

            return self._embed_legacy_endpoint(client, texts)
        except httpx.HTTPStatusError as exc:
            if not allow_model_fallback or exc.response is None:
                raise
            if not self._switch_to_available_model(client, exc.response):
                raise
            return self._embed_texts_with_client(client, texts, allow_model_fallback=False)

    def _embed_batch_endpoint(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        endpoint = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": texts}
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
            self._endpoint_mode = "batch"
            return embeddings

        if "embedding" in data and len(texts) == 1:
            self._endpoint_mode = "batch"
            return [data["embedding"]]

        raise RuntimeError(f"Unexpected Ollama embed response shape: {data}")

    def _embed_legacy_endpoint(self, client: httpx.Client, texts: list[str]) -> list[list[float]]:
        endpoint = f"{self.base_url}/api/embeddings"
        out: list[list[float]] = []
        for text in texts:
            payload = {"model": self.model, "prompt": text}
            response = client.post(endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            embedding = data.get("embedding")
            if embedding is None:
                raise RuntimeError(f"Unexpected Ollama legacy embed response shape: {data}")
            out.append(embedding)

        self._endpoint_mode = "legacy"
        return out

    def _switch_to_available_model(self, client: httpx.Client, response: httpx.Response) -> bool:
        if response.status_code != 404:
            return False

        message = self._extract_error_message(response).lower()
        if "model" not in message or "not found" not in message:
            return False

        tags_response = client.get(f"{self.base_url}/api/tags")
        if tags_response.is_error:
            return False

        payload = tags_response.json()
        models = [
            str(model.get("name", "")).strip()
            for model in payload.get("models", [])
            if str(model.get("name", "")).strip()
        ]
        if not models:
            return False
        if self.model in models:
            return False

        preferred = next((name for name in models if "embed" in name.lower()), models[0])
        self.model = preferred
        return True

    @staticmethod
    def _extract_error_message(response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("error")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()

        text = response.text.strip()
        if text:
            return text
        return f"HTTP {response.status_code}"

    def embed_in_batches(self, texts: Iterable[str], batch_size: int = 16) -> list[list[float]]:
        source = list(texts)
        out: list[list[float]] = []
        for i in range(0, len(source), batch_size):
            out.extend(self.embed_texts(source[i : i + batch_size]))
        return out
