"""Pseudo-relevance feedback: refine a Query from its own top results."""

from __future__ import annotations

from copilot.contracts import Candidate, Query


def expand(query: Query, feedback_docs: list[Candidate], *, k_docs: int = 3) -> Query:
    """Return a new Query whose `dense_vec_override` is the centroid of the top
    `k_docs` fused candidates' embeddings. `free_text` is left unchanged."""
    raise NotImplementedError
