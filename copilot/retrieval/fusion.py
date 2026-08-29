"""Combine lexical and dense candidate lists into one ranking."""

from __future__ import annotations

from copilot.contracts import Candidate, Query


def fuse(
    bm25_hits: list[Candidate],
    dense_hits: list[Candidate],
    query: Query,
    *,
    top_k: int = 50,
) -> list[Candidate]:
    """Merge the two lists (dedupe on `parent_asin`). Blend weight is a function
    of `query.intent_p_buying`. Returns best-first, len <= top_k, with
    `fused_score` and 1-based `fused_rank` set on each Candidate."""
    raise NotImplementedError
