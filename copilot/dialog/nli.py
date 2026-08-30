from __future__ import annotations

import os

from ..models import NliCrossEncoder


HYPOTHESES: dict[str, dict[str, str]] = {
    "intent": {
        "buying": "The shopper wants to buy a specific item and has a firm requirement.",
        "browsing": "The shopper is just browsing and has not decided what they want.",
    },
    "use_case": {
        "running": "This is for running or jogging.",
        "gym": "This is for the gym or working out.",
        "hiking": "This is for hiking or the outdoors.",
        "work": "This is for the office or for work.",
        "wedding": "This is for a wedding or a formal occasion.",
        "winter": "This is for cold or winter weather.",
        "summer": "This is for hot or summer weather.",
        "travel": "This is for travelling.",
        "rain": "This is for rain or wet weather.",
    },
    "no_preference": {
        "yes": "The shopper has no preference and wants the assistant to decide.",
    },
    "hard_reset": {
        "yes": "The shopper is discarding an earlier preference and replacing it with a new one.",
    },
    "dissatisfied": {
        "yes": "The shopper is unhappy with the options that have been shown so far.",
    },
}

class ZeroShotNliScorer:
    def __init__(self, backend: "NliCrossEncoder | None" = None, *, model_name: str | None = None) -> None:
        if backend is None:
            backend = NliCrossEncoder(model_name) if model_name else NliCrossEncoder()
        self._backend = backend

    @classmethod
    def maybe(cls) -> "ZeroShotNliScorer | None":
        if os.environ.get("COPILOT_NLI", "off").strip().lower() in ("", "0", "off", "false", "no"):
            return None
        try:
            scorer = cls()
            scorer._backend.entails_batch([("probe", "probe")])
            return scorer
        except Exception:
            return None

    def _score(self, premise: str, hyps: dict[str, str]) -> dict[str, float]:
        try:
            scores = self._backend.entails_batch([(premise, h) for h in hyps.values()])
        except Exception:
            return {}
        return dict(zip(hyps.keys(), scores)) if scores else {}

    def intent_p_buying(self, message: str) -> float | None:
        s = self._score(message, HYPOTHESES["intent"])
        if not s:
            return None
        b, br = s["buying"], s["browsing"]
        return b / (b + br) if (b + br) > 1e-6 else None

    def use_case(self, message: str, *, tau: float = 0.6) -> str | None:
        s = self._score(message, HYPOTHESES["use_case"])
        if not s:
            return None
        label, score = max(s.items(), key=lambda kv: kv[1])
        return label if score >= tau else None

    def flag(self, job: str, message: str, *, tau: float = 0.65) -> bool:
        s = self._score(message, HYPOTHESES[job])
        return bool(s) and s.get("yes", 0.0) >= tau