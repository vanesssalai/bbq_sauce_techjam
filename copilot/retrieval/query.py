"""Turn session state into the single Query object the retrieval stack consumes."""

from __future__ import annotations

from copilot.contracts import ParsedTurn, Query, SessionState, Slot


def active_slots(state: SessionState) -> dict[str, Slot]:
    """Slots currently in force: newest value per attribute, overrides applied,
    relaxed attributes dropped."""
    raise NotImplementedError


def build_query(state: SessionState, parsed: ParsedTurn) -> Query:
    """Assemble the Query for this turn. `free_text` is the one string sent to
    both BM25 MATCH and the dense encoder; negations stay out of it."""
    raise NotImplementedError
