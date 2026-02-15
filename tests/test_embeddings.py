from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from cv_rag.embeddings import OllamaEmbedClient

# ===========================================================================
# Helpers
# ===========================================================================


def _make_response(*, status_code: int = 200, json_data: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.text = text
    resp.is_error = status_code >= 400
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("no json")
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    return resp


# ===========================================================================
# embed_texts — basic
# ===========================================================================


def test_embed_texts_empty_returns_empty() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")
    assert client.embed_texts([]) == []


def test_embed_texts_batch_endpoint_happy_path() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    response = _make_response(json_data={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.return_value = response
        MockClient.return_value = mock_http

        result = client.embed_texts(["hello", "world"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    assert client._endpoint_mode == "batch"


def test_embed_texts_single_text_embedding_key() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    response = _make_response(json_data={"embedding": [0.5, 0.6]})

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.return_value = response
        MockClient.return_value = mock_http

        result = client.embed_texts(["single"])

    assert result == [[0.5, 0.6]]


# ===========================================================================
# embed_texts — legacy fallback
# ===========================================================================


def test_embed_texts_falls_back_to_legacy_on_404() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    batch_404 = _make_response(status_code=404, json_data={"error": "not found"})
    legacy_ok = _make_response(json_data={"embedding": [0.7, 0.8]})

    call_count = {"n": 0}

    def fake_post(url, **kwargs):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        if url.endswith("/api/embed"):
            return batch_404
        return legacy_ok

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.side_effect = fake_post
        MockClient.return_value = mock_http

        result = client.embed_texts(["text"])

    assert result == [[0.7, 0.8]]
    assert client._endpoint_mode == "legacy"


# ===========================================================================
# embed_texts — endpoint mode persistence
# ===========================================================================


def test_endpoint_mode_persists_across_calls() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")
    client._endpoint_mode = "legacy"

    legacy_resp = _make_response(json_data={"embedding": [0.1]})

    urls_called: list[str] = []

    def fake_post(url, **kwargs):  # type: ignore[no-untyped-def]
        urls_called.append(url)
        return legacy_resp

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.side_effect = fake_post
        MockClient.return_value = mock_http

        client.embed_texts(["a"])

    # Should only call legacy endpoint, not try batch first
    assert all("/api/embeddings" in u for u in urls_called)


# ===========================================================================
# embed_texts — model fallback
# ===========================================================================


def test_model_fallback_on_not_found() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "missing-model")

    error_404 = _make_response(status_code=404, json_data={"error": "model 'missing-model' not found"})
    tags_resp = _make_response(json_data={"models": [{"name": "nomic-embed-text"}, {"name": "llama2"}]})
    legacy_success = _make_response(json_data={"embedding": [0.1, 0.2]})

    call_count = {"post": 0}

    # Flow: batch 404 → legacy 404 → switch model → legacy success
    def fake_post(url, **kwargs):  # type: ignore[no-untyped-def]
        call_count["post"] += 1
        if call_count["post"] <= 2:
            return error_404
        return legacy_success

    def fake_get(url, **kwargs):  # type: ignore[no-untyped-def]
        return tags_resp

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.side_effect = fake_post
        mock_http.get.side_effect = fake_get
        MockClient.return_value = mock_http

        result = client.embed_texts(["test"])

    assert result == [[0.1, 0.2]]
    assert client.model == "nomic-embed-text"


# ===========================================================================
# embed_texts — error cases
# ===========================================================================


def test_embed_texts_count_mismatch_raises() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    response = _make_response(json_data={"embeddings": [[0.1]]})  # 1 embedding for 2 texts

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.return_value = response
        MockClient.return_value = mock_http

        with pytest.raises(RuntimeError, match="mismatched"):
            client.embed_texts(["a", "b"])


def test_embed_texts_unexpected_shape_raises() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    response = _make_response(json_data={"something_else": True})

    with patch("httpx.Client") as MockClient:
        mock_http = MagicMock()
        mock_http.__enter__ = MagicMock(return_value=mock_http)
        mock_http.__exit__ = MagicMock(return_value=False)
        mock_http.post.return_value = response
        MockClient.return_value = mock_http

        with pytest.raises(RuntimeError, match="Unexpected"):
            client.embed_texts(["a", "b"])


# ===========================================================================
# embed_in_batches
# ===========================================================================


def test_embed_in_batches_splits_correctly() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")

    call_args: list[list[str]] = []

    def tracking_embed(texts: list[str]) -> list[list[float]]:
        call_args.append(texts)
        return [[float(i)] for i in range(len(texts))]

    client.embed_texts = tracking_embed  # type: ignore[assignment]

    result = client.embed_in_batches(["a", "b", "c", "d", "e"], batch_size=2)

    assert len(call_args) == 3  # 2 + 2 + 1
    assert len(call_args[0]) == 2
    assert len(call_args[1]) == 2
    assert len(call_args[2]) == 1
    assert len(result) == 5


def test_embed_in_batches_empty() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")
    assert client.embed_in_batches([]) == []


# ===========================================================================
# _extract_error_message
# ===========================================================================


def test_extract_error_message_json_error() -> None:
    resp = _make_response(json_data={"error": "model not found"})
    assert OllamaEmbedClient._extract_error_message(resp) == "model not found"


def test_extract_error_message_text_fallback() -> None:
    resp = _make_response(text="Bad Request")
    assert OllamaEmbedClient._extract_error_message(resp) == "Bad Request"


def test_extract_error_message_http_status_fallback() -> None:
    resp = _make_response(status_code=500, text="")
    assert OllamaEmbedClient._extract_error_message(resp) == "HTTP 500"


# ===========================================================================
# _switch_to_available_model
# ===========================================================================


def test_switch_model_non_404_returns_false() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "test-model")
    resp = _make_response(status_code=500, text="server error")
    mock_http = MagicMock()
    assert client._switch_to_available_model(mock_http, resp) is False


def test_switch_model_no_model_error_returns_false() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "test-model")
    resp = _make_response(status_code=404, json_data={"error": "endpoint not found"})
    mock_http = MagicMock()
    assert client._switch_to_available_model(mock_http, resp) is False


def test_switch_model_prefers_embed_model() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "missing-model")
    resp = _make_response(status_code=404, json_data={"error": "model 'missing-model' not found"})

    tags_resp = _make_response(json_data={"models": [{"name": "llama2"}, {"name": "nomic-embed-text"}]})

    mock_http = MagicMock()
    mock_http.get.return_value = tags_resp

    assert client._switch_to_available_model(mock_http, resp) is True
    assert client.model == "nomic-embed-text"


def test_switch_model_falls_back_to_first() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "missing-model")
    resp = _make_response(status_code=404, json_data={"error": "model 'missing-model' not found"})

    tags_resp = _make_response(json_data={"models": [{"name": "llama2"}, {"name": "mistral"}]})

    mock_http = MagicMock()
    mock_http.get.return_value = tags_resp

    assert client._switch_to_available_model(mock_http, resp) is True
    assert client.model == "llama2"


def test_switch_model_empty_models_returns_false() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "missing-model")
    resp = _make_response(status_code=404, json_data={"error": "model 'missing-model' not found"})

    tags_resp = _make_response(json_data={"models": []})

    mock_http = MagicMock()
    mock_http.get.return_value = tags_resp

    assert client._switch_to_available_model(mock_http, resp) is False


def test_switch_model_already_available_returns_false() -> None:
    client = OllamaEmbedClient("http://localhost:11434", "nomic-embed-text")
    resp = _make_response(status_code=404, json_data={"error": "model 'nomic-embed-text' not found"})

    tags_resp = _make_response(json_data={"models": [{"name": "nomic-embed-text"}]})

    mock_http = MagicMock()
    mock_http.get.return_value = tags_resp

    assert client._switch_to_available_model(mock_http, resp) is False
