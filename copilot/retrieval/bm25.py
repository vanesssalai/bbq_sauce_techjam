"""Lexical (BM25) retrieval over the frozen catalog."""

from __future__ import annotations

from pathlib import Path

from copilot.contracts import Candidate, Query


class Bm25Index:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)

    def search(self, query: Query, *, limit: int = 100) -> list[Candidate]:
        """Top `limit` lexical hits for `query.free_text`, hard filters from
        `query.slots` applied. Each Candidate gets `bm25_score` and
        `filter_match` set."""
        raise NotImplementedError
