from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from cv_rag.storage.qdrant import QdrantStore, VectorPoint

# ---------------------------------------------------------------------------
# _normalize_point_id
# ---------------------------------------------------------------------------


def test_normalize_point_id_integer_passthrough() -> None:
    assert QdrantStore._normalize_point_id(42) == 42


def test_normalize_point_id_valid_uuid_passthrough() -> None:
    uid = str(uuid.uuid4())
    assert QdrantStore._normalize_point_id(uid) == uid


def test_normalize_point_id_string_to_uuid5() -> None:
    result = QdrantStore._normalize_point_id("2104.00680:0")
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "2104.00680:0"))
    assert result == expected


def test_normalize_point_id_deterministic() -> None:
    a = QdrantStore._normalize_point_id("same-key")
    b = QdrantStore._normalize_point_id("same-key")
    assert a == b


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


def _make_store(mock_client: MagicMock) -> QdrantStore:
    store = object.__new__(QdrantStore)
    store.client = mock_client
    store.collection_name = "test_col"
    return store


def test_ensure_collection_creates_when_missing() -> None:
    client = MagicMock()
    collections_resp = MagicMock()
    collections_resp.collections = []
    client.get_collections.return_value = collections_resp

    store = _make_store(client)
    store.ensure_collection(vector_size=768)

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["vectors_config"].size == 768


def test_ensure_collection_skips_when_size_matches() -> None:
    client = MagicMock()
    col = MagicMock()
    col.name = "test_col"
    collections_resp = MagicMock()
    collections_resp.collections = [col]
    client.get_collections.return_value = collections_resp

    info = MagicMock()
    info.config.params.vectors.size = 768
    client.get_collection.return_value = info

    store = _make_store(client)
    store.ensure_collection(vector_size=768)

    client.create_collection.assert_not_called()


def test_ensure_collection_raises_on_size_mismatch() -> None:
    client = MagicMock()
    col = MagicMock()
    col.name = "test_col"
    collections_resp = MagicMock()
    collections_resp.collections = [col]
    client.get_collections.return_value = collections_resp

    info = MagicMock()
    info.config.params.vectors.size = 384
    client.get_collection.return_value = info

    store = _make_store(client)

    with pytest.raises(RuntimeError, match="vector size 384"):
        store.ensure_collection(vector_size=768)


def test_ensure_collection_handles_dict_vectors() -> None:
    client = MagicMock()
    col = MagicMock()
    col.name = "test_col"
    collections_resp = MagicMock()
    collections_resp.collections = [col]
    client.get_collections.return_value = collections_resp

    inner_vec = MagicMock()
    inner_vec.size = 768
    # Use a real dict so hasattr(vectors, "size") is False → enters dict branch
    info = MagicMock()
    info.config.params.vectors = {"default": inner_vec}
    client.get_collection.return_value = info

    store = _make_store(client)
    store.ensure_collection(vector_size=768)

    client.create_collection.assert_not_called()


def test_delete_collection_if_exists_returns_false_when_missing() -> None:
    client = MagicMock()
    collections_resp = MagicMock()
    collections_resp.collections = []
    client.get_collections.return_value = collections_resp
    store = _make_store(client)

    deleted = store.delete_collection_if_exists()

    assert deleted is False
    client.delete_collection.assert_not_called()


def test_delete_collection_if_exists_deletes_and_returns_true() -> None:
    client = MagicMock()
    col = MagicMock()
    col.name = "test_col"
    collections_resp = MagicMock()
    collections_resp.collections = [col]
    client.get_collections.return_value = collections_resp
    store = _make_store(client)

    deleted = store.delete_collection_if_exists()

    assert deleted is True
    client.delete_collection.assert_called_once_with(collection_name="test_col")


# ---------------------------------------------------------------------------
# upsert
# ---------------------------------------------------------------------------


def test_upsert_empty_list_noop() -> None:
    client = MagicMock()
    store = _make_store(client)
    store.upsert([])
    client.upsert.assert_not_called()


def test_upsert_single_batch() -> None:
    client = MagicMock()
    store = _make_store(client)

    points = [
        VectorPoint(point_id="chunk-0", vector=[0.1, 0.2], payload={"text": "hello"}),
        VectorPoint(point_id="chunk-1", vector=[0.3, 0.4], payload={"text": "world"}),
    ]
    store.upsert(points, batch_size=64)

    assert client.upsert.call_count == 1
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == "test_col"
    assert kwargs["wait"] is True
    assert len(kwargs["points"]) == 2


def test_upsert_multi_batch() -> None:
    client = MagicMock()
    store = _make_store(client)

    points = [
        VectorPoint(point_id=f"chunk-{i}", vector=[float(i)], payload={})
        for i in range(5)
    ]
    store.upsert(points, batch_size=2)

    assert client.upsert.call_count == 3  # 2 + 2 + 1


def test_upsert_preserves_payload() -> None:
    client = MagicMock()
    store = _make_store(client)

    payload = {"chunk_id": "c1", "arxiv_id": "2104.00680", "text": "content"}
    points = [VectorPoint(point_id="c1", vector=[0.1], payload=payload)]
    store.upsert(points)

    _, kwargs = client.upsert.call_args
    point_struct = kwargs["points"][0]
    assert point_struct.payload == payload


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_extracts_payload() -> None:
    client = MagicMock()
    store = _make_store(client)

    item = MagicMock()
    item.payload = {
        "chunk_id": "2104.00680:0",
        "arxiv_id": "2104.00680",
        "title": "LoFTR",
        "section_title": "Abstract",
        "text": "Feature matching.",
    }
    item.score = 0.92
    item.id = "some-uuid"
    client.search.return_value = [item]

    results = store.search(query_vector=[0.1, 0.2], limit=5)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "2104.00680:0"
    assert results[0]["arxiv_id"] == "2104.00680"
    assert results[0]["title"] == "LoFTR"
    assert results[0]["score"] == pytest.approx(0.92)


def test_search_empty_results() -> None:
    client = MagicMock()
    store = _make_store(client)
    client.search.return_value = []

    results = store.search(query_vector=[0.1], limit=5)
    assert results == []


def test_search_falls_back_to_query_points() -> None:
    client = MagicMock()
    store = _make_store(client)

    client.search.side_effect = AttributeError("no search method")

    item = MagicMock()
    item.payload = {"chunk_id": "c1", "arxiv_id": "a1", "title": "T", "section_title": "S", "text": "t"}
    item.score = 0.8
    item.id = "uuid"
    query_result = MagicMock()
    query_result.points = [item]
    client.query_points.return_value = query_result

    results = store.search(query_vector=[0.1], limit=3)

    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"
    client.query_points.assert_called_once()


def test_search_handles_missing_payload_fields() -> None:
    client = MagicMock()
    store = _make_store(client)

    item = MagicMock()
    item.payload = {}
    item.score = 0.5
    item.id = "fallback-id"
    client.search.return_value = [item]

    results = store.search(query_vector=[0.1], limit=1)

    assert results[0]["chunk_id"] == "fallback-id"
    assert results[0]["arxiv_id"] == ""
    assert results[0]["title"] == ""
    assert results[0]["text"] == ""
