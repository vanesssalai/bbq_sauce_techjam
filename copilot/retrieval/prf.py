"""Cross-turn Pseudo-Relevance Feedback (Rocchio): sharpen the query from its own
top results before the final rank.

Flag-gated in the agent; the orchestrator re-runs bm25 + dense + fuse with the
refined query. A coherence guard makes it a no-op when the top results disagree,
so a bad turn cannot drag the query off course.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace

import numpy as np

from copilot.contracts import Candidate, Query
from copilot.retrieval.bm25 import _terms
from copilot.retrieval.dense import DenseIndex

_NEG_BAND = (30, 60)          # rank window treated as "non-relevant" for Rocchio
_MAX_EXPANSION_TERMS = 10

# catalog IDF, built once (keyed by the catalog object's id)
_IDF: dict[str, float] | None = None
_IDF_FOR: int | None = None


def _catalog_idf(catalog: dict[str, Candidate]) -> dict[str, float]:
    global _IDF, _IDF_FOR
    if _IDF is not None and _IDF_FOR == id(catalog):
        return _IDF
    n_docs = len(catalog) or 1
    doc_freq: dict[str, int] = {}
    for cand in catalog.values():
        for tok in set(_terms(cand.search_text)):
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    _IDF = {tok: math.log(n_docs / (1 + df)) for tok, df in doc_freq.items()}
    _IDF_FOR = id(catalog)
    return _IDF


def _norm(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n else vec


def _rows_for(pids, dense_index: DenseIndex) -> list[int]:
    return [dense_index.id_to_row[p] for p in pids if p in dense_index.id_to_row]


def _mean_pairwise_cosine(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return 0.0
    sims = mat @ mat.T
    iu = np.triu_indices(mat.shape[0], k=1)
    return float(sims[iu].mean())


def _expansion_terms(pids, catalog, idf, already: set[str]) -> list[str]:
    local_tf: dict[str, int] = {}
    for pid in pids:
        cand = catalog.get(pid)
        if not cand:
            continue
        for tok in _terms(cand.search_text):
            local_tf[tok] = local_tf.get(tok, 0) + 1
    scored = [
        (tok, tf * idf.get(tok, 0.0))
        for tok, tf in local_tf.items()
        if tok not in already and len(tok) > 2 and not tok.isdigit()
    ]
    scored.sort(key=lambda kv: -kv[1])
    return [tok for tok, _ in scored[:_MAX_EXPANSION_TERMS]]


def refine_query(
    query: Query,
    fused: list[tuple[str, float]],
    dense_index: DenseIndex,
    catalog: dict[str, Candidate],
    *,
    top_k: int = 8,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.15,
    coherence_floor: float = 0.35,
) -> Query:
    """Return a NEW `Query` with augmented `free_text` and `dense_vec_override`
    set from the top-`top_k` fused results -- or `query` unchanged if those
    results are incoherent. Never touches `query.slots` / `query.negations`.
    """
    pos_pids = [pid for pid, _ in fused[:top_k]]
    pos_rows = _rows_for(pos_pids, dense_index)
    if len(pos_rows) < 2:
        return query

    pos = dense_index.matrix[pos_rows]                       # (k, dim), L2-normed rows
    coherence = _mean_pairwise_cosine(pos)
    if coherence < coherence_floor:
        return query                                        # results disagree -> don't trust them

    if query.dense_vec_override is not None:
        q_vec = _norm(np.asarray(query.dense_vec_override, dtype="float32"))
    else:
        q_vec = _norm(dense_index.encoder.encode(query.free_text, is_query=True).astype("float32"))

    centroid_pos = _norm(pos.mean(axis=0))
    neg_rows = _rows_for([pid for pid, _ in fused[_NEG_BAND[0]:_NEG_BAND[1]]], dense_index)
    centroid_neg = (
        _norm(dense_index.matrix[neg_rows].mean(axis=0))
        if neg_rows else np.zeros_like(q_vec)
    )

    new_vec = _norm(alpha * q_vec + (beta * coherence) * centroid_pos - gamma * centroid_neg)

    already = set(_terms(query.free_text))
    extra = _expansion_terms(pos_pids, catalog, _catalog_idf(catalog), already)
    new_text = (query.free_text + " " + " ".join(extra)).strip() if extra else query.free_text

    return replace(query, free_text=new_text, dense_vec_override=new_vec.tolist())
