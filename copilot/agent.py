from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from .contracts import SessionState, UserProfile
from .dialog.intent import build_intent_scorer
from .dialog.nli import ZeroShotNliScorer
from .dialog.nlu import extract_slots_and_intent
from .dialog.semantic_slots import build_semantic_resolver
from .dialog.state_machine import active_slots, apply_turn, record_shown

TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


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

    def _search(self, query_text: str, top_k: int) -> list[dict]:
        terms = list(dict.fromkeys(_terms(query_text)))[:40]
        if not terms:
            return []
        expression = " OR ".join(f'"{t}"' for t in terms)
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, top_k),
        ).fetchall()
        return [{"parent_asin": str(row[0])} for row in rows]

    @staticmethod
    def _accumulated_query(state: SessionState) -> str:
        parts: list[str] = []
        for attr, slot in active_slots(state).items():
            if attr not in ("price_min", "price_max"):
                parts.append(slot.value)
        parts.extend(state.disclosed_phrases)
        if "category" not in state.slots and state.raw_history:
            parts.append(state.raw_history[0][1])
        return " ".join(parts)

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
        apply_turn(state, parsed)

        query_text = self._accumulated_query(state) or user_message
        recommendations = self._search(query_text, top_k)
        record_shown(state, [r["parent_asin"] for r in recommendations])

        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
