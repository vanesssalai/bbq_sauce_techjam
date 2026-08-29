"""Second-stage cross-encoder reranking."""

from __future__ import annotations

from copilot.contracts import Candidate, Query, RankResult
from copilot.models import CrossEncoder


def rank(
    query: Query,
    candidates: list[Candidate],
    *,
    top_k: int = 10,
    encoder: CrossEncoder | None = None,
) -> RankResult:
    """Cross-encoder rerank of the fused pool against `query.free_text`. Sets
    `rank_score` on each returned Candidate and fills `score_gap` / `why`."""
    raise NotImplementedError
