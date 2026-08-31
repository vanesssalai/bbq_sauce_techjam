from __future__ import annotations

import os
from dataclasses import dataclass, replace

from copilot.contracts import Candidate, Query
from copilot.retrieval import fusion
from copilot.retrieval.bm25 import Bm25Index
from copilot.retrieval.dense import DenseIndex
from copilot.retrieval.filters import apply_filters, apply_filters_with_relaxation
from copilot.models import BiEncoder

last_relaxation: tuple[str, str | None] | None = None


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


_FILTER_RESTRICT = _flag("COPILOT_FILTER_RESTRICT")
_NO_DENSE = not _flag("COPILOT_USE_DENSE")
_RETRIEVE_LIMIT = _envi("COPILOT_RETRIEVE_LIMIT", 400)


@dataclass
class RetrievalIndexes:
    bm25: Bm25Index
    dense: DenseIndex
    encoder: BiEncoder


def retrieve(
    query: Query,
    catalog: dict[str, Candidate],
    indexes: RetrievalIndexes,
    *,
    limit: int = _RETRIEVE_LIMIT,
    channel_limit: int = 600,
) -> list[Candidate]:
    global last_relaxation

    survivors = apply_filters(list(catalog.values()), query.slots, query.negations)
    survivor_ids = {c.parent_asin for c in survivors}
    if survivors:
        last_relaxation = None
    else:
        _, last_relaxation = apply_filters_with_relaxation(
            list(catalog.values()), query.slots, query.negations
        )

    allowed_ids: set[str] | None = survivor_ids if (_FILTER_RESTRICT and survivors) else None

    lexical = indexes.bm25.search(
        query.free_text, allowed_ids=allowed_ids, limit=channel_limit,
        phrases=getattr(query, "phrases", None),
    )
    if _NO_DENSE:
        dense = []
        fused = fusion.fuse({"bm25": lexical}, {"bm25": 1.0}, limit=limit)
    else:
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
            filter_match=(parent_asin in survivor_ids),
        )
        for rank, (parent_asin, fused_score) in enumerate(fused, start=1)
    ]
