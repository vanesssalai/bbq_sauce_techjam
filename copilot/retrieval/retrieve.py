"""First-stage retrieval orchestrator: filter -> bm25 + dense -> fuse.

`retrieve()` is Dev R's deliverable to Dev K: given a `Query`, the in-memory
catalog, and the built indexes, it returns ~`limit` fresh `Candidate` copies,
best-first by fused score, each with `bm25_score`, `dense_score`, `fused_score`,
`fused_rank` (1-indexed) and `filter_match=True` set.

PRF (`prf.refine_query`) is an optional extra pass the agent runs between
`retrieve()` and `rank()`; it is not called from here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from copilot.contracts import Candidate, Query
from copilot.retrieval import fusion
from copilot.retrieval.bm25 import Bm25Index
from copilot.retrieval.dense import DenseIndex
from copilot.retrieval.filters import apply_filters, apply_filters_with_relaxation
from copilot.models import BiEncoder

# When the hard filters wipe out the pool, retrieve() falls back to the whole
# catalog so a turn never returns zero recommendations, and records the
# relaxation hint here for clarify.py to pick up. Single-threaded eval only;
# revisit as an explicit return channel at integration.
last_relaxation: tuple[str, str | None] | None = None


@dataclass
class RetrievalIndexes:
    """Built once in `agent.reset()`, owned by Dev R. `encoder` is shared with
    the NLU intent scorer / semantic resolver."""

    bm25: Bm25Index
    dense: DenseIndex
    encoder: BiEncoder


def retrieve(
    query: Query,
    catalog: dict[str, Candidate],
    indexes: RetrievalIndexes,
    *,
    limit: int = 200,
    channel_limit: int = 400,
) -> list[Candidate]:
    global last_relaxation

    survivors = apply_filters(list(catalog.values()), query.slots, query.negations)
    if survivors:
        last_relaxation = None
        allowed_ids: set[str] | None = {c.parent_asin for c in survivors}
    else:
        _, last_relaxation = apply_filters_with_relaxation(
            list(catalog.values()), query.slots, query.negations
        )
        allowed_ids = None  # search the whole catalog rather than return nothing

    lexical = indexes.bm25.search(query.free_text, allowed_ids=allowed_ids, limit=channel_limit)
    dense = indexes.dense.search(query, allowed_ids=allowed_ids, limit=channel_limit)
    fused = fusion.fuse(
        {"bm25": lexical, "dense": dense},
        fusion.weights(query),
        limit=limit,
    )

    lexical_scores = dict(lexical)
    dense_scores = dict(dense)
    return [
        replace(
            catalog[parent_asin],
            bm25_score=lexical_scores.get(parent_asin, 0.0),
            dense_score=dense_scores.get(parent_asin, 0.0),
            fused_score=fused_score,
            fused_rank=rank,
            filter_match=True,
        )
        for rank, (parent_asin, fused_score) in enumerate(fused, start=1)
    ]
