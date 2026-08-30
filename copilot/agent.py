from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from .catalog import load_catalog
from .contracts import Candidate, SessionState, UserProfile
from .dialog.nlu import extract_slots_and_intent
from .dialog.state_machine import (
    apply_turn,
    note_clarification,
    record_shown,
    should_ask,
)
from .ranking.rank import rank
from .retrieval import retrieve as _retrieve_mod
from .retrieval.query import build_query
from .retrieval.retrieve import RetrievalIndexes, retrieve

_MAX_TURNS = 10
_DENSE_NPY = Path("data/dense_embeddings.npy")
_DENSE_META = Path("data/embedding_meta.json")


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "on", "true", "yes")


PRF_ON = _flag("COPILOT_PRF")
TOURNAMENT_ON = _flag("COPILOT_TOURNAMENT")
CE_OFF = _flag("COPILOT_NO_CROSS_ENCODER")


try:
    _RANK_TOPK = int(os.environ.get("COPILOT_RANK_TOPK", "200"))
except ValueError:
    _RANK_TOPK = 200

_ASK_PRIORITY = ("feature", "material", "use_case", "color", "style", "budget", "size", "brand")
_ASK_TEMPLATES = {
    "feature": "Is there a specific feature that matters most here?",
    "material": "Any material or fabric you're set on?",
    "use_case": "What will you mainly use it for?",
    "color": "Any colour you have in mind?",
    "style": "What style are you going for?",
    "budget": "Roughly what's your budget?",
    "size": "What size do you need?",
    "brand": "Any brand you prefer?",
    "other": "Anything else that matters for this?",
}

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


def _parse_profile(user_profile: dict | None) -> UserProfile | None:
    if not user_profile:
        return None
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
        self.catalog = load_catalog(self.catalog_path)

        self.indexes: RetrievalIndexes | None = None
        self.cross_encoder = None
        self.rank_weights = None
        self._intent_scorer = None
        self._sem_resolver = None
        self._nli = None
        self._models_ready = False
        self._last_relaxation: tuple[str, str | None] | None = None

        self._sessions: dict[str, SessionState] = {}

    def _ensure_ready(self) -> None:
        if self._models_ready:
            return
        self._models_ready = True
        try:
            self._build_indexes()
        except Exception:
            self.indexes = None
        self._build_optional_models()

    def _build_indexes(self) -> None:
        from .models import BiEncoder
        from .retrieval.bm25 import Bm25Index
        from .retrieval.dense import DenseIndex

        encoder = BiEncoder()
        bm25 = Bm25Index(self.catalog_path)
        if _DENSE_NPY.is_file() and _DENSE_META.is_file():
            try:
                dense = DenseIndex.load(_DENSE_NPY, _DENSE_META, encoder)
            except Exception:
                dense = DenseIndex.build(self.catalog, encoder)
        else:  # slower startup fallback — run scripts/build_artifacts.py to skip this
            dense = DenseIndex.build(self.catalog, encoder)
        self.indexes = RetrievalIndexes(bm25=bm25, dense=dense, encoder=encoder)

    def _build_optional_models(self) -> None:
        if _flag("COPILOT_NO_NLU_MODELS"):
            if not CE_OFF:
                try:
                    from .models import CrossEncoder

                    ce = CrossEncoder()
                    ce.score("probe", ["probe"])
                    self.cross_encoder = ce
                except Exception:
                    self.cross_encoder = None
            return

        shared = None
        if self.indexes is not None:
            try:
                from .models import Encoder

                shared = Encoder(self.indexes.encoder)
            except Exception:
                shared = None

        if not _flag("COPILOT_NO_INTENT_SCORER"):
            try:
                from .dialog.intent import build_intent_scorer

                self._intent_scorer = build_intent_scorer(shared)
            except Exception:
                self._intent_scorer = None

        if _flag("COPILOT_SEM_SLOTS"):
            try:
                from .dialog.semantic_slots import build_semantic_resolver

                self._sem_resolver = build_semantic_resolver(shared)
            except Exception:
                self._sem_resolver = None

        try:
            from .dialog.nli import ZeroShotNliScorer

            self._nli = ZeroShotNliScorer.maybe()
        except Exception:
            self._nli = None

        if not CE_OFF:
            try:
                from .models import CrossEncoder

                ce = CrossEncoder()
                ce.score("probe", ["probe"])  # force the lazy load; degrade if it fails
                self.cross_encoder = ce
            except Exception:
                self.cross_encoder = None

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._ensure_ready()
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            user_profile=_parse_profile(user_profile or {}),
        )

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            return self._respond(session_id, user_message, turn, top_k)
        except Exception:
            return {
                "message": "Here are some options.",
                "ask_attribute": None,
                "recommendations": self._fallback(user_message, top_k),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }

    def _respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            self.reset(session_id, {})
            state = self._sessions[session_id]

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
            
            if _flag("COPILOT_PIVOT_CLEAR_PHRASES"):
                state.disclosed_phrases.clear()

        apply_turn(state, parsed)

        query = build_query(state, parsed)
        candidates = self._retrieve(query)

        result = rank(
            candidates,
            query,
            state,
            top_k=max(top_k, _RANK_TOPK),
            cross_encoder=self.cross_encoder,
            weights=self.rank_weights,
            use_tournament=TOURNAMENT_ON and self.cross_encoder is not None,
        )

        recommendations = self._order(result.ranked, state, top_k)
        record_shown(state, [r["parent_asin"] for r in recommendations])

        message, ask_attribute = self._dialogue(state, parsed, turn)
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def _retrieve(self, query) -> list[Candidate]:
        self._last_relaxation = None
        if self.indexes is None:
            return self._popularity_pool()

        try:
            cands = retrieve(query, self.catalog, self.indexes)
        except Exception:
            return self._popularity_pool()
        self._last_relaxation = getattr(_retrieve_mod, "last_relaxation", None)

        if PRF_ON and cands:
            try:
                from .retrieval.prf import refine_query

                refined = refine_query(
                    query,
                    [(c.parent_asin, c.fused_score or 0.0) for c in cands],
                    self.indexes.dense,
                    self.catalog,
                )
                if refined is not query:
                    cands = retrieve(refined, self.catalog, self.indexes)
            except Exception:
                pass

        return cands or self._popularity_pool()

    def _popularity_pool(self, size: int = 200) -> list[Candidate]:
        pool = sorted(
            self.catalog.values(),
            key=lambda c: (c.rating_number, c.average_rating),
            reverse=True,
        )[:size]
        return [
            replace(c, fused_score=1.0 / (i + 1), fused_rank=i + 1)
            for i, c in enumerate(pool)
        ]

    def _order(self, ranked: list[Candidate], state: SessionState, top_k: int) -> list[dict]:
        seen: set[str] = set()
        deduped: list[Candidate] = []
        for c in ranked:
            if c.parent_asin in seen:
                continue
            seen.add(c.parent_asin)
            deduped.append(c)
        unseen = [c for c in deduped if c.parent_asin not in state.shown_asins]
        ordered = unseen if len(unseen) >= top_k else (
            unseen + [c for c in deduped if c.parent_asin in state.shown_asins]
        )
        return [{"parent_asin": c.parent_asin} for c in ordered[:top_k]]

    def _dialogue(self, state: SessionState, parsed, turn: int) -> tuple[str, str | None]:
        relax = self._last_relaxation
        if relax and turn < _MAX_TURNS and not state.no_preference:
            state.pending_relaxation = relax
            return _relaxation_prompt(*relax), None

        if parsed.is_override or parsed.is_hard_reset:
            return "Got it — here are matches for that instead.", None

        for attr in _ASK_PRIORITY:
            if attr in state.asked_attributes or attr in state.no_preference:
                continue
            if not should_ask(state, attr):
                continue
            note_clarification(state)
            state.pending_ask_attribute = attr
            state.asked_attributes.add(attr)
            if attr == "other":
                state.other_ask_count += 1
            return _ASK_TEMPLATES.get(attr, f"Could you tell me more about {attr}?"), attr

        return "Here are the closest matches I found.", None

    def _fallback(self, user_message: str, top_k: int) -> list[dict]:
        try:
            if self.indexes is not None:
                hits = self.indexes.bm25.search(user_message, limit=top_k)
                if hits:
                    return [{"parent_asin": pid} for pid, _ in hits[:top_k]]
        except Exception:
            pass
        return [{"parent_asin": pid} for pid in list(self.catalog)[:top_k]]
