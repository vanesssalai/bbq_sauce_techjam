"""Dense (bi-encoder) retrieval: cosine similarity over a precomputed matrix.

Prefer `DenseIndex.load(...)` against the artifact from
`scripts/build_artifacts.py`; `DenseIndex.build(catalog, encoder)` is the slow
startup fallback when the artifact is missing or stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from copilot.contracts import Candidate, Query
from copilot.models import BiEncoder


def _l2(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


class DenseIndex:
    def __init__(self, matrix: np.ndarray, catalog_ids: list[str], encoder: BiEncoder) -> None:
        self.matrix = np.ascontiguousarray(matrix, dtype="float32")  # (N, dim), rows L2-normed
        self.catalog_ids = catalog_ids
        self.id_to_row = {pid: i for i, pid in enumerate(catalog_ids)}
        self.encoder = encoder

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(
        cls,
        npy_path: str | Path,
        meta_path: str | Path,
        encoder: BiEncoder | None = None,
    ) -> "DenseIndex":
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        matrix = np.load(npy_path).astype("float32")
        ids = meta["parent_asins"]
        if len(ids) != matrix.shape[0]:
            raise ValueError("embedding_meta parent_asins length != matrix rows")
        return cls(matrix, ids, encoder or BiEncoder())

    @classmethod
    def build(
        cls,
        catalog: dict[str, Candidate],
        encoder: BiEncoder | None = None,
    ) -> "DenseIndex":
        encoder = encoder or BiEncoder()
        ids = list(catalog)
        vecs = encoder.encode([catalog[pid].search_text for pid in ids], normalize=True)
        return cls(np.asarray(vecs, dtype="float32"), ids, encoder)

    # -- query --------------------------------------------------------------
    def _query_vec(self, query: Query) -> np.ndarray:
        if query.dense_vec_override is not None:
            return _l2(np.asarray(query.dense_vec_override, dtype="float32"))
        return _l2(self.encoder.encode(query.free_text, is_query=True).astype("float32"))

    def search(
        self,
        query: Query,
        allowed_ids: set[str] | None = None,
        limit: int = 400,
    ) -> list[tuple[str, float]]:
        q_vec = self._query_vec(query)
        n = self.matrix.shape[0]

        if allowed_ids is not None and len(allowed_ids) < 0.3 * n:
            rows = [self.id_to_row[i] for i in allowed_ids if i in self.id_to_row]
            if not rows:
                return []
            scores = self.matrix[rows] @ q_vec
            take = min(limit, len(rows))
            top = np.argpartition(-scores, take - 1)[:take]
            top = top[np.argsort(-scores[top])]
            return [(self.catalog_ids[rows[i]], float(scores[i])) for i in top]

        scores = self.matrix @ q_vec
        if allowed_ids is not None:
            mask = np.ones(n, dtype=bool)
            for i in allowed_ids:
                row = self.id_to_row.get(i)
                if row is not None:
                    mask[row] = False
            scores = scores.copy()
            scores[mask] = -np.inf

        take = min(limit, n)
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        return [(self.catalog_ids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]
