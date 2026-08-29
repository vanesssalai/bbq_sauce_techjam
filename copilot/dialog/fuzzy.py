from __future__ import annotations

from typing import Iterable


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = f" {text.lower().strip()} "
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def trigram_jaccard(a: str, b: str) -> float:
    ga, gb = char_ngrams(a), char_ngrams(b)
    if not ga or not gb:
        return 0.0
    inter = len(ga & gb)
    return inter / (len(ga) + len(gb) - inter)


def best_fuzzy_match(
    token: str, vocab: Iterable[str], *, threshold: float = 0.55, min_len: int = 5
) -> tuple[str, float] | None:
    if len(token) < min_len:
        return None
    best: str | None = None
    best_score = threshold
    for cand in vocab:
        score = trigram_jaccard(token, cand)
        if score > best_score:
            best, best_score = cand, score
    return (best, best_score) if best is not None else None
