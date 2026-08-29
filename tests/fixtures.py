"""Shared test fixtures: a hand-built fused `Candidate` pool (descending
`fused_score`, 1-based `fused_rank`) so `rank()` and `retrieve()` can be tested
independently.
"""

from __future__ import annotations

import copy

from copilot.contracts import Candidate

TARGET_ASIN = "B000000003"


def make_candidate(
    parent_asin: str,
    title: str,
    *,
    fused_score: float | None = None,
    fused_rank: int | None = None,
    **overrides,
) -> Candidate:
    candidate = Candidate(parent_asin=parent_asin, title=title)
    if fused_score is not None:
        candidate.fused_score = fused_score
    if fused_rank is not None:
        candidate.fused_rank = fused_rank
    for name, value in overrides.items():
        if not hasattr(candidate, name):
            raise AttributeError(f"Candidate has no field {name!r}")
        setattr(candidate, name, value)
    return candidate


_FUSED_CANDIDATES: list[Candidate] = [
    make_candidate(
        "B000000001", "Cotton Crew T-Shirt, White",
        fused_score=0.92, fused_rank=1,
        brand="Hanes", categories=["Clothing", "Tops", "T-Shirts"],
        colors=["white"], material="cotton", department="men",
        sizes=["S", "M", "L", "XL"], price=12.99,
        average_rating=4.4, rating_number=1820,
        bm25_score=8.1, dense_score=0.71,
        search_text="cotton crew neck t-shirt white soft breathable everyday basic",
    ),
    make_candidate(
        "B000000002", "Polyester Running Shorts, Black",
        fused_score=0.81, fused_rank=2,
        brand="Nike", categories=["Clothing", "Activewear", "Shorts"],
        colors=["black"], material="polyester", department="men",
        sizes=["M", "L", "XL"], price=24.0,
        average_rating=4.6, rating_number=940,
        bm25_score=6.7, dense_score=0.66,
        search_text="polyester running shorts black moisture wicking gym training",
    ),
    make_candidate(
        "B000000003", "Linen Button-Down Shirt, Navy",
        fused_score=0.74, fused_rank=3,
        brand="J. Crew", categories=["Clothing", "Tops", "Shirts"],
        colors=["navy", "blue"], material="linen", department="men",
        sizes=["S", "M", "L"], price=68.0,
        average_rating=4.2, rating_number=310,
        bm25_score=5.9, dense_score=0.63,
        search_text="linen button down shirt navy lightweight summer breathable",
    ),
    make_candidate(
        "B000000004", "Wool Blend Socks, 3-Pack Gray",
        fused_score=0.55, fused_rank=4,
        brand="Darn Tough", categories=["Clothing", "Socks"],
        colors=["gray"], material="wool", department="unisex",
        sizes=["M", "L"], price=19.5,
        average_rating=4.8, rating_number=5600,
        bm25_score=4.1, dense_score=0.48,
        search_text="merino wool blend socks gray cushioned hiking warm",
    ),
    make_candidate(
        "B000000005", "Leather Chelsea Boots, Brown",
        fused_score=0.41, fused_rank=5,
        brand="Thursday", categories=["Shoes", "Boots"],
        colors=["brown"], material="leather", department="men",
        sizes=["9", "10", "11"], price=199.0,
        average_rating=4.5, rating_number=760,
        bm25_score=3.3, dense_score=0.39,
        search_text="leather chelsea boots brown goodyear welt ankle dress",
    ),
    make_candidate(
        "B000000006", "Silk Scarf, Floral Pink",
        fused_score=0.28, fused_rank=6,
        brand="Echo", categories=["Accessories", "Scarves"],
        colors=["pink"], material="silk", department="women",
        price=45.0, average_rating=4.1, rating_number=95,
        bm25_score=2.0, dense_score=0.25,
        search_text="silk scarf floral pink lightweight accessory",
    ),
]


def fused_candidates() -> list[Candidate]:
    """Fresh deep copy of the fused pool, safe to mutate."""
    return [copy.deepcopy(candidate) for candidate in _FUSED_CANDIDATES]
