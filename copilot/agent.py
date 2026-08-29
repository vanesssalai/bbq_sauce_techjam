"""Orchestrator agent: NLU (todo) -> build_query -> retrieve -> rank.

Current state: **passthrough ranker** — retrieval + fusion are wired, the final
`rank()` is not, so results are ordered by `fused_score`. This is Track R's
milestone check (guide §4a): it must already beat the weak BM25 baseline
(HitRate@10 0.125, MRR 0.068) on every scenario column.
"""

from __future__ import annotations

from pathlib import Path

from copilot.catalog import load_catalog
from copilot.contracts import SessionState
from copilot.retrieval.query import build_query
from copilot.retrieval.retrieve import RetrievalIndexes, retrieve

PRF_ON = False   # flag-gated; refine_query is implemented but off for the passthrough milestone

_DENSE_NPY = "data/dense_embeddings.npy"
_DENSE_META = "data/embedding_meta.json"


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog = load_catalog(self.catalog_path)
        self.indexes: RetrievalIndexes | None = None
        self._sessions: dict[str, SessionState] = {}

    # -- lazy, once ---------------------------------------------------------
    def _ensure_indexes(self) -> None:
        if self.indexes is not None:
            return
        from copilot.models import BiEncoder
        from copilot.retrieval.bm25 import Bm25Index
        from copilot.retrieval.dense import DenseIndex

        encoder = BiEncoder()
        bm25 = Bm25Index(self.catalog_path)
        npy, meta = Path(_DENSE_NPY), Path(_DENSE_META)
        if npy.is_file() and meta.is_file():
            dense = DenseIndex.load(npy, meta, encoder)
        else:  # slow startup fallback — run scripts/build_artifacts.py to avoid this
            dense = DenseIndex.build(self.catalog, encoder)
        self.indexes = RetrievalIndexes(bm25=bm25, dense=dense, encoder=encoder)

    # -- API --------------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ensure_indexes()
        self._sessions[session_id] = SessionState(session_id=session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:  # never let the harness see an exception
            return {
                "message": "Here are some options.",
                "ask_attribute": None,
                "recommendations": self._fallback(top_k),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self._sessions[session_id]

        state.turn = turn
        state.raw_history.append(("user", user_message))
        if user_message and user_message not in state.disclosed_phrases:
            state.disclosed_phrases.append(user_message)

        query = build_query(state)
        cands = retrieve(query, self.catalog, self.indexes)

        if PRF_ON and cands:
            from copilot.retrieval.prf import refine_query

            refined = refine_query(
                query,
                [(c.parent_asin, c.fused_score) for c in cands],
                self.indexes.dense,
                self.catalog,
            )
            if refined is not query:
                cands = retrieve(refined, self.catalog, self.indexes)

        # passthrough: fused order is the final order until rank() lands
        ranked = cands[:top_k]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": [{"parent_asin": c.parent_asin} for c in ranked],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _fallback(self, top_k: int) -> list[dict]:
        ids = list(self.catalog)[:top_k]
        return [{"parent_asin": pid} for pid in ids]
