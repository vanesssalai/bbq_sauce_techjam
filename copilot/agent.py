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
import json
import re
import sqlite3
from dataclasses import replace

from .catalog import normalize_row
from .contracts import ASK_ATTRIBUTE_TO_SLOTS, Candidate, SessionState, UserProfile
from .dialog.intent import build_intent_scorer
from .dialog.nli import ZeroShotNliScorer
from .dialog.nlu import extract_slots_and_intent
from .dialog.semantic_slots import build_semantic_resolver
from .dialog.state_machine import (
    apply_turn,
    effective_confidence,
    note_clarification,
    record_shown,
)
from .retrieval.filters import apply_filters_with_relaxation
from .retrieval.query import build_query


_ASK_PRIORITY = ("material", "color", "budget", "use_case", "style", "size", "brand")
_ASK_TEMPLATES = {
    "material": "Any material or fabric you prefer?",
    "color": "Do you have a colour in mind?",
    "budget": "What's your budget for this?",
    "use_case": "What will you mainly use it for?",
    "style": "What style are you going for?",
    "size": "What size do you need?",
    "brand": "Any particular brand you like?",
    "other": "Anything else that matters for this?",
}
_MAX_DISTINCT_ASKS = 6
_MAX_OTHER_ASKS = 2
_PINNED = 0.6
_MAX_TURNS = 10
_POOL_MULTIPLIER = 40  # BM25 candidates pulled per requested result, before filtering
_POOL_MIN = 400

_RELAX_LABELS = {
    "size": "size", "department": "department", "category": "category",
    "price_min": "minimum price", "price_max": "budget",
}


def _relaxation_prompt(attr: str, new_value: str | None) -> str:
    if attr == "price_max" and new_value:
        return f"I couldn't find anything in that budget. Include options up to about ${float(new_value):.0f}?"
    if attr == "price_min" and new_value:
        return f"Nothing matched above that price. Include options from about ${float(new_value):.0f}?"
    return f"I couldn't find an exact match on {_RELAX_LABELS.get(attr, attr)}. Want me to loosen that?"


PRF_ON = False   # flag-gated; refine_query is implemented but off for the passthrough milestone

_DENSE_NPY = "data/dense_embeddings.npy"
_DENSE_META = "data/embedding_meta.json"


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.catalog = load_catalog(self.catalog_path)
        self.indexes: RetrievalIndexes | None = None
        self._sessions: dict[str, SessionState] = {}

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

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ensure_indexes()
        self._sessions[session_id] = SessionState(session_id=session_id)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:  
            return {
                "message": "Here are some options.",
                "ask_attribute": None,
                "recommendations": self._fallback(top_k),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

def _parse_profile(user_profile: dict) -> UserProfile | None:
    try:
        return UserProfile(
            purchase_frequency=str(user_profile.get("purchase_frequency", "")),
            average_prior_rating=float(user_profile.get("average_prior_rating") or 0.0),
            rating_style=str(user_profile.get("rating_style", "")),
            preference_tags=list(user_profile.get("preference_tags") or []),
            summary=str(user_profile.get("summary", "")),
        )
    except Exception:
        return None


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, SessionState] = {}
        self._catalog: dict[str, Candidate] = {}
        self._build_index()
        self._nlu_ready = False
        self._intent_scorer = None
        self._sem_resolver = None
        self._nli = None

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, ...]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                candidate = normalize_row(product)
                if candidate is not None:
                    self._catalog[candidate.parent_asin] = candidate
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def _ensure_nlu_models(self) -> None:
        if self._nlu_ready:
            return
        self._nlu_ready = True
        try:
            self._intent_scorer = build_intent_scorer()
        except Exception:
            self._intent_scorer = None
        try:
            self._sem_resolver = build_semantic_resolver()
        except Exception:
            self._sem_resolver = None
        self._nli = ZeroShotNliScorer.maybe()

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ensure_nlu_models()
        self._sessions[session_id] = SessionState(
            session_id=session_id, user_profile=_parse_profile(user_profile or {})
        )

    def _bm25_pool(self, query_text: str, limit: int) -> list[str]:
        terms = list(dict.fromkeys(_terms(query_text)))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{t}"' for t in terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _search(self, query_text: str, top_k: int, exclude: set[str] | None = None) -> list[dict]:
        asins = self._bm25_pool(query_text, max(top_k * 15, 150))
        exclude = exclude or set()
        unseen = [a for a in asins if a not in exclude]
        ranked = unseen if len(unseen) >= top_k else unseen + [a for a in asins if a in exclude]
        return [{"parent_asin": a} for a in ranked[:top_k]]

    def _next_ask_attribute(self, state: SessionState, turn: int) -> str | None:
        if turn >= _MAX_TURNS:
            return None
        if len(state.asked_attributes) < _MAX_DISTINCT_ASKS:
            for attr in _ASK_PRIORITY:
                if attr in state.asked_attributes or attr in state.no_preference:
                    continue
                slot_keys = ASK_ATTRIBUTE_TO_SLOTS.get(attr, [])
                if any(
                    (slot := state.slots.get(key)) is not None
                    and slot.source in ("explicit", "clarification_answer")
                    and effective_confidence(slot, state.turn) >= _PINNED
                    for key in slot_keys
                ):
                    continue
                return attr
        if state.other_ask_count < _MAX_OTHER_ASKS and not state.has_pivoted:
            return "other"
        return None

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            return {
                "message": "Here are some options.",
                "ask_attribute": None,
                "recommendations": self._search(user_message, top_k),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
<<<<<<< HEAD
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
=======
            state = self._sessions[session_id] = SessionState(session_id=session_id)

        parsed = extract_slots_and_intent(
            user_message,
            turn,
            pending_ask_attribute=state.pending_ask_attribute,
            prior_track=state.current_track,
            intent_scorer=self._intent_scorer,
            semantic_resolver=self._sem_resolver,
            nli=self._nli,
            prior_slots={k: s.value for k, s in state.slots.items()},
        )

        if parsed.is_hard_reset and parsed.slots:
            parsed = replace(parsed, is_hard_reset=False)
        if (parsed.is_override or parsed.is_hard_reset) and not state.has_pivoted:
            state.has_pivoted = True
            state.shown_asins.clear()

        apply_turn(state, parsed)

        query = build_query(state, parsed)
        pool_asins = self._bm25_pool(query.free_text or user_message, max(top_k * _POOL_MULTIPLIER, _POOL_MIN))

        pool = [replace(self._catalog[a]) for a in pool_asins if a in self._catalog]

        survivors, relaxation = apply_filters_with_relaxation(pool, query.hard_slots, query.negations)
        ranked = pool

        unseen = [c for c in ranked if c.parent_asin not in state.shown_asins]
        ordered = unseen if len(unseen) >= top_k else unseen + [c for c in ranked if c.parent_asin in state.shown_asins]
        recommendations = [{"parent_asin": c.parent_asin} for c in ordered[:top_k]]
        record_shown(state, [r["parent_asin"] for r in recommendations])

        ask_attribute = None
        if not survivors and relaxation and turn < _MAX_TURNS:
           
            state.pending_relaxation = relaxation
            message = _relaxation_prompt(*relaxation)
        else:
            ask_attribute = self._next_ask_attribute(state, turn)
            if ask_attribute:
                note_clarification(state)
                state.pending_ask_attribute = ask_attribute
                if ask_attribute == "other":
                    state.other_ask_count += 1
                else:
                    state.asked_attributes.add(ask_attribute)
                message = _ASK_TEMPLATES.get(ask_attribute, f"Could you tell me more about {ask_attribute}?")
            else:
                message = "Here are the closest matches I found."

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
>>>>>>> a2f8e915626ba38da1bcbeea66253f07b3ab2de5
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _fallback(self, top_k: int) -> list[dict]:
        ids = list(self.catalog)[:top_k]
        return [{"parent_asin": pid} for pid in ids]
