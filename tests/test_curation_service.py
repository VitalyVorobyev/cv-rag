from __future__ import annotations

from cv_rag.curation.service import (
    CurateOptions,
    CurateThresholds,
    curate_corpus,
    is_curatable_paper_id,
)


class _FakeSQLiteStore:
    def __init__(self, arxiv_ids: list[str]) -> None:
        self._arxiv_ids = arxiv_ids
        self.upserted_rows: list[dict[str, object]] = []

    def list_paper_arxiv_ids(self, limit: int | None = None) -> list[str]:
        if limit is None:
            return list(self._arxiv_ids)
        return list(self._arxiv_ids[:limit])

    def get_paper_metric_timestamps(self, arxiv_ids: list[str]) -> dict[str, int]:
        _ = arxiv_ids
        return {}

    def upsert_paper_metrics(self, metric_rows: list[dict[str, object]]) -> None:
        self.upserted_rows.extend(metric_rows)

    def get_paper_metrics_tier_distribution(self) -> dict[int, int]:
        return {2: len(self.upserted_rows)}


class _FakeS2Client:
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    def get_papers_batch(
        self,
        list_of_arxiv_ids: list[str],
        fields: tuple[str, ...],
    ) -> list[dict[str, object] | None]:
        _ = fields
        self.batch_calls.append(list(list_of_arxiv_ids))
        return [
            {
                "citationCount": 12,
                "year": 2021,
                "venue": "CVPR",
                "publicationTypes": ["Conference"],
            }
            for _ in list_of_arxiv_ids
        ]

    def get_paper(self, arxiv_id: str, fields: tuple[str, ...]) -> dict[str, object] | None:
        _ = (arxiv_id, fields)
        return None


def test_is_curatable_paper_id_accepts_arxiv_and_doi_identifiers() -> None:
    assert is_curatable_paper_id("2104.00680")
    assert is_curatable_paper_id("2104.00680v2")
    assert is_curatable_paper_id("cs.CV/9901001")
    assert is_curatable_paper_id("ARXIV:2104.00680")
    assert is_curatable_paper_id("doi:10.1186/s40537-019-0197-0")
    assert is_curatable_paper_id("10.1186/s40537-019-0197-0")

    assert not is_curatable_paper_id("")
    assert not is_curatable_paper_id("url:abc123")
    assert not is_curatable_paper_id("https://example.org/paper.pdf")


def test_curate_corpus_skips_non_arxiv_ids_by_default() -> None:
    sqlite_store = _FakeSQLiteStore(
        arxiv_ids=[
            "2104.00680",
            "doi:10.1186/s40537-019-0197-0",
            "url:44eda04d3c406276f3837d7dd66c139bcf0baea3525718cfbbfd6ef7d2f7711b",
        ]
    )
    s2_client = _FakeS2Client()

    result = curate_corpus(
        sqlite_store=sqlite_store,  # type: ignore[arg-type]
        s2_client=s2_client,  # type: ignore[arg-type]
        venue_whitelist={"CVPR"},
        options=CurateOptions(
            refresh_days=30,
            limit=None,
            skip_non_arxiv=True,
            thresholds=CurateThresholds(),
        ),
    )

    assert result.total_ids == 3
    assert result.to_refresh == 2
    assert result.updated == 2
    assert result.skipped == 1
    assert result.skipped_non_curatable == 1
    assert s2_client.batch_calls == [["2104.00680", "doi:10.1186/s40537-019-0197-0"]]
    assert len(sqlite_store.upserted_rows) == 2
