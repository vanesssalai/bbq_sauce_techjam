"""Load the frozen catalog JSONL into `Candidate` objects, once, in memory.

`load_catalog(path) -> dict[parent_asin, Candidate]` is the shared read model for
retrieval, filtering, and ranking. `dense_text(row)` is the single string builder
used both here (for `Candidate.search_text`) and by `scripts/build_artifacts.py`
(for the embedding matrix) so the two never drift.

Structured colour / material / size are sparse in the source data (~5% have a
`details.Color`, ~4% a `details.Material`), so those are scraped from the
title+features corpus with the same regexes the evaluator uses. ~87% of rows
carry `details.Department`; ~21% carry a `price`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from copilot.contracts import Candidate

_COLOR_RE = re.compile(
    r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I
)
_MATERIAL_RE = re.compile(
    r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I
)
_SIZE_SPLIT_RE = re.compile(r"[,/|]| and ")

_DEPARTMENT_ALIASES = {
    "womens": "women", "women": "women", "woman": "women",
    "mens": "men", "men": "men", "man": "men",
    "boys": "boys", "girls": "girls", "kids": "kids", "unisex-adult": "unisex",
    "unisex adult": "unisex", "unisex": "unisex", "baby": "baby",
}


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items() if v not in (None, "", []))
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value if v not in (None, ""))
    return str(value)


_DENSE_TEXT_MAX_WORDS = 320  # bge-small truncates at 512 tokens; the long tail of a
                             # feature list is low-value and slows encoding.


def dense_text(row: dict) -> str:
    """title + features + flattened details, capped. No `description` (noisy). Tunable."""
    title = str(row.get("title") or "")
    features = _as_text(row.get("features"))
    details = _as_text(row.get("details"))
    joined = " ".join(part for part in (title, features, details) if part).strip()
    words = joined.split()
    return " ".join(words[:_DENSE_TEXT_MAX_WORDS]) if len(words) > _DENSE_TEXT_MAX_WORDS else joined


def _normalize_department(raw: object) -> str | None:
    if not raw:
        return None
    key = str(raw).strip().lower()
    return _DEPARTMENT_ALIASES.get(key, key or None)


def _scrape_colors(corpus: str) -> list[str]:
    seen: list[str] = []
    for match in _COLOR_RE.findall(corpus):
        value = match.lower()
        if value not in seen:
            seen.append(value)
    return seen


def _scrape_material(corpus: str, details: dict) -> str | None:
    explicit = details.get("Material")
    if explicit:
        return str(explicit).strip().lower()
    match = _MATERIAL_RE.search(corpus)
    return match.group(1).lower() if match else None


def _scrape_sizes(details: dict) -> list[str]:
    for key, value in details.items():
        if "size" in str(key).lower() and value not in (None, "", []):
            return [s.strip() for s in _SIZE_SPLIT_RE.split(str(value)) if s.strip()]
    return []


def _to_candidate(row: dict) -> Candidate:
    details = row.get("details") or {}
    corpus = f"{row.get('title') or ''} {_as_text(row.get('features'))}"
    price = row.get("price")
    return Candidate(
        parent_asin=str(row["parent_asin"]),
        title=str(row.get("title") or ""),
        brand=(row.get("store") or details.get("Brand") or details.get("Manufacturer") or None),
        categories=[str(c) for c in (row.get("categories") or [])],
        colors=_scrape_colors(corpus),
        sizes=_scrape_sizes(details),
        material=_scrape_material(corpus, details),
        department=_normalize_department(details.get("Department")),
        price=float(price) if isinstance(price, (int, float)) else None,
        average_rating=float(row.get("average_rating") or 0.0),
        rating_number=int(row.get("rating_number") or 0),
        search_text=dense_text(row),
    )


def load_catalog(path: str | Path) -> dict[str, Candidate]:
    """Parse `catalog.jsonl` into `{parent_asin: Candidate}`, source line order preserved."""
    catalog: dict[str, Candidate] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            candidate = _to_candidate(json.loads(line))
            catalog[candidate.parent_asin] = candidate
    return catalog
