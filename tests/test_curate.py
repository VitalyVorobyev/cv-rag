from __future__ import annotations

from cv_rag.curate import CurateThresholds, compute_tier_and_score


def test_compute_tier_and_score_assigns_tier0_for_top_venue_and_high_citations() -> None:
    thresholds = CurateThresholds()
    tier, score, is_peer_reviewed = compute_tier_and_score(
        citation_count=350,
        year=2021,
        venue="CVPR",
        publication_types=["Conference"],
        venue_whitelist={"CVPR", "ICCV"},
        thresholds=thresholds,
        current_year=2026,
    )

    assert tier == 0
    assert score > 0
    assert is_peer_reviewed == 1


def test_compute_tier_and_score_assigns_tier1_for_peer_reviewed_mid_citations() -> None:
    thresholds = CurateThresholds(
        tier0_min_citations=200,
        tier0_min_cpy=30.0,
        tier1_min_citations=20,
        tier1_min_cpy=3.0,
    )
    tier, score, is_peer_reviewed = compute_tier_and_score(
        citation_count=35,
        year=2022,
        venue="Unknown Workshop",
        publication_types=["JournalArticle"],
        venue_whitelist={"CVPR"},
        thresholds=thresholds,
        current_year=2026,
    )

    assert tier == 1
    assert score > 0
    assert is_peer_reviewed == 1


def test_compute_tier_and_score_assigns_tier2_for_non_peer_reviewed_low_citations() -> None:
    thresholds = CurateThresholds()
    tier, score, is_peer_reviewed = compute_tier_and_score(
        citation_count=2,
        year=2025,
        venue="arXiv",
        publication_types=["Preprint"],
        venue_whitelist={"CVPR"},
        thresholds=thresholds,
        current_year=2026,
    )

    assert tier == 2
    assert score >= 0
    assert is_peer_reviewed == 0
