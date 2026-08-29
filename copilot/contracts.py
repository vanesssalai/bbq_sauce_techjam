from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

SlotAttribute = Literal[
    "color", "material", "size", "department", "category",
    "price_min", "price_max", "brand",
]

AskAttribute = Literal[
    "category", "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "other",
]

HARD_FILTER_ATTRS: set[str] = {"size", "department", "category", "price_min", "price_max"}
SOFT_BOOST_ATTRS: set[str] = {"color", "material", "brand"}

SLOT_TO_ASK_ATTRIBUTE: dict[str, str] = {
    "color": "color", "material": "material", "size": "size",
    "category": "category", "brand": "brand",
    "price_min": "budget", "price_max": "budget",
    "department": "category",
}

ASK_ATTRIBUTE_TO_SLOTS: dict[str, list[str]] = {
    "color": ["color"], "material": ["material"], "size": ["size"],
    "category": ["category", "department"], "brand": ["brand"],
    "budget": ["price_min", "price_max"],
    "style": [], "feature": [], "use_case": [], "other": [],
}

@dataclass
class Candidate:
    parent_asin: str
    title: str
    brand: str | None = None
    categories: list[str] = field(default_factory=list)
    colors: list[str] = field(default_factory=list)
    sizes: list[str] = field(default_factory=list)
    material: str | None = None
    department: str | None = None
    price: float | None = None
    average_rating: float = 0.0
    rating_number: int = 0
    search_text: str = ""
    bm25_score: float = 0.0
    filter_match: bool = False
    dense_score: float = 0.0
    fused_score: float | None = None
    fused_rank: int | None = None
    rank_score: float | None = None

@dataclass
class Slot:
    value: str
    confidence: float
    turn_set: int
    source: Literal["explicit", "clarification_answer", "inferred", "llm"]

@dataclass
class ParsedTurn:
    raw_text: str
    turn: int
    intent: Literal["buying", "browsing"]
    intent_confidence: float
    is_override: bool
    overridden_attrs: list[str]
    slots: dict[str, Slot]
    soft_tags: list[str]
    negated_values: dict[str, list[str]]
    answered_ask_attribute: str | None
    is_dissatisfied: bool = False
    dissatisfaction_attribute: str | None = None
    disclosed_phrases: list[str] = field(default_factory=list)
    is_no_preference: bool = False
    no_preference_attribute: str | None = None
    is_hard_reset: bool = False
    intent_p_buying: float = 0.5
    rewritten_query: str | None = None
    intent_tier: str = "none"
    is_affirmation: bool = False
    is_rejection: bool = False

@dataclass
class UserProfile:
    purchase_frequency: str
    average_prior_rating: float
    rating_style: str
    preference_tags: list[str]
    summary: str

@dataclass
class SessionState:
    session_id: str
    turn: int = 0
    user_profile: UserProfile | None = None
    current_track: Literal["buying", "browsing"] | None = None
    slots: dict[str, Slot] = field(default_factory=dict)
    negated_values: dict[str, list[str]] = field(default_factory=dict)
    pending_ask_attribute: str | None = None
    asked_attributes: set[str] = field(default_factory=set)
    other_ask_count: int = 0
    has_pivoted: bool = False
    relaxed_attrs: list[str] = field(default_factory=list)
    distilled_summary: str = ""
    raw_history: list[tuple[str, str]] = field(default_factory=list)
    shown_asins: set[str] = field(default_factory=set)
    no_preference: set[str] = field(default_factory=set)
    disclosed_phrases: list[str] = field(default_factory=list)
    clarify_count: int = 0
    intent_p_buying: float = 0.5
    pending_relaxation: tuple[str, str | None] | None = None

@dataclass
class TurnResult:
    message: str
    ask_attribute: str | None = None
    recommendations: list[Candidate] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})

    def to_response(self, top_k: int = 10) -> dict[str, Any]:
        return {
            "message": self.message,
            "ask_attribute": self.ask_attribute,
            "recommendations": [
                {"parent_asin": c.parent_asin, **({"score": c.rank_score} if c.rank_score is not None else {})}
                for c in self.recommendations[:top_k]
            ],
            "usage": self.usage,
        }

@dataclass
class Query:
    free_text: str
    slots: dict[str, Slot]
    negations: dict[str, list[str]]
    intent_p_buying: float
    track: Literal["buying", "browsing"]
    turn: int
    dense_vec_override: list[float] | None = None

@dataclass
class RankResult:
    ranked: list[Candidate]
    score_gap: float
    why: dict[str, str]