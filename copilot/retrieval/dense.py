"""Dense (bi-encoder) retrieval over the frozen catalog."""

from __future__ import annotations

from pathlib import Path

from copilot.contracts import Candidate, Query
from copilot.models import BiEncoder


class DenseIndex:
    def __init__(self, catalog_path: str | Path, encoder: BiEncoder | None = None) -> None:
        self.catalog_path = Path(catalog_path)
        self.encoder = encoder or BiEncoder()

    def search(self, query: Query, *, limit: int = 100) -> list[Candidate]:
        """Top `limit` cosine neighbours. Uses `query.dense_vec_override` when
        set (PRF), otherwise encodes `query.free_text`. Sets `dense_score`."""
        raise NotImplementedError
