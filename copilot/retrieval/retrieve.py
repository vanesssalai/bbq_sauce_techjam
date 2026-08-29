"""First-stage retrieval orchestrator: bm25 + dense -> fuse -> optional PRF."""

from __future__ import annotations

from pathlib import Path

from copilot.contracts import Candidate, Query


class Retriever:
    def __init__(
        self,
        catalog_path: str | Path,
        *,
        use_dense: bool = True,
        use_prf: bool = True,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.use_dense = use_dense
        self.use_prf = use_prf

    def retrieve(self, query: Query, *, top_k: int = 50) -> list[Candidate]:
        """Fused candidate pool, best-first, len <= top_k, `fused_rank` set.
        Runs one PRF round when `use_prf` and the top result looks weak."""
        raise NotImplementedError
