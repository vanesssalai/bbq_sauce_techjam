from __future__ import annotations

from copilot.contracts import Candidate, Query, Slot, SessionState


def make_candidate(
    parent_asin: str,
    title: str,
    *,
    price: float | None = None,
    brand: str | None = None,
    categories: list[str] | None = None,
    colors: list[str] | None = None,
    material: str | None = None,
    department: str | None = None,
    average_rating: float = 4.0,
    rating_number: int = 100,
    bm25_score: float = 0.0,
    dense_score: float = 0.0,
    fused_score: float | None = None,
    fused_rank: int | None = None,
) -> Candidate:
    return Candidate(
        parent_asin=parent_asin,
        title=title,
        brand=brand,
        categories=categories or [],
        colors=colors or [],
        material=material,
        department=department,
        price=price,
        average_rating=average_rating,
        rating_number=rating_number,
        bm25_score=bm25_score,
        dense_score=dense_score,
        fused_score=fused_score,
        fused_rank=fused_rank,
        filter_match=True,
    )


def make_fixture_candidates() -> list[Candidate]:
    """~5 hand-built candidates with preset scores, for testing rank() in isolation."""
    return [
        make_candidate("A1", "Leather Ankle Boots Brown", price=45.0, material="leather",
                        colors=["brown"], categories=["shoes", "boots"],
                        bm25_score=0.85, dense_score=0.80,
                        fused_score=0.90, fused_rank=1),
        make_candidate("A2", "Suede Ankle Boots Tan", price=120.0, material="suede",
                        colors=["tan"], categories=["shoes", "boots"],
                        bm25_score=0.55, dense_score=0.70,
                        fused_score=0.75, fused_rank=2),
        make_candidate("A3", "Canvas Sneakers White", price=30.0, material="canvas",
                        colors=["white"], categories=["shoes", "sneakers"],
                        bm25_score=0.70, dense_score=0.40,
                        fused_score=0.60, fused_rank=3),
        make_candidate("A4", "Leather Chelsea Boots Black", price=95.0, material="leather",
                        colors=["black"], categories=["shoes", "boots"],
                        bm25_score=0.40, dense_score=0.65,
                        fused_score=0.55, fused_rank=4),
        make_candidate("A5", "Rain Boots Yellow", price=25.0, material="rubber",
                        colors=["yellow"], categories=["shoes", "boots"],
                        bm25_score=0.30, dense_score=0.35,
                        fused_score=0.40, fused_rank=5),
        # A6: weak on bm25/dense/fused but the cross-encoder alone will (falsely) love it,
        # since fused_score/fused_rank were never actually computed from these numbers.
        # Lets us see the tournament pull a lone outlier CE score back toward consensus.
        make_candidate("A6", "Leather Ankle Boots Espresso", price=50.0, material="leather",
                        colors=["brown"], categories=["shoes", "boots"],
                        bm25_score=0.20, dense_score=0.15,
                        fused_score=0.10, fused_rank=6),
    ]


def make_fixture_query(**overrides) -> Query:
    defaults = dict(
        free_text="leather ankle boots",
        slots={"material": Slot(value="leather", confidence=0.9, turn_set=1, source="explicit")},
        negations={},
        intent_p_buying=0.8,
        track="buying",
        turn=1,
    )
    defaults.update(overrides)
    return Query(**defaults)


def make_fixture_session(**overrides) -> SessionState:
    defaults = dict(session_id="test-session-1", turn=1, current_track="buying")
    defaults.update(overrides)
    return SessionState(**defaults)
