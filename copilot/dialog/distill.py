"""Compress conversation history into a short summary for context construction."""

from __future__ import annotations

from copilot.contracts import SessionState


def distill(state: SessionState) -> str:
    """Short natural-language recap of `state.raw_history` (stored back on
    `state.distilled_summary`), so later turns keep the prompt small."""
    raise NotImplementedError
