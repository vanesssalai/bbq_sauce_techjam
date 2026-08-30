
# to load the catalog.jsonl and extract attributes 
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from .contracts import Candidate

log = logging.getLogger(__name__)


_COLOR_WORDS = [
    "black", "white", "grey", "gray", "red", "blue", "navy", "green",
    "olive", "brown", "tan", "beige", "khaki", "pink", "purple", "yellow",
    "orange", "maroon", "burgundy", "teal", "gold", "golden", "silver",
    "ivory", "cream", "clear", "turquoise", "rainbow", "rose gold",
    "multicolor", "multicolored", "multicoloured", "multi-color", "multi",
]


_MATERIAL_WORDS = [
    "cotton", "leather", "wool", "silk", "polyester", "denim", "linen",
    "suede", "cashmere", "nylon", "spandex", "velvet", "canvas", "rayon",
    "fleece", "chiffon", "metal", "stainless steel", "faux leather",
    "plastic", "synthetic", "wood", "silicone", "crystal", "glass",
    "sterling silver", "gemstone", "acrylic", "stone", "alloy", "brass",
    "aluminum", "aluminium", "polyurethane", "rhinestone", "resin", "rubber",
]


_SIZE_LETTER_RE = re.compile(r"(?<!['’])\b(XX?S|S|M|L|XX?L|XXX?L|[2-5]X)\b", re.IGNORECASE)
_SIZE_NUMERIC_RE = re.compile(r"\bsize[s]?\s*[:\-]?\s*(\d{1,2}(?:\.\d)?)\b", re.IGNORECASE)

_SIZE_WORD_MAP: dict[str, str] = {
    "xx-small": "XXS", "xx small": "XXS", "extra extra small": "XXS",
    "x-small": "XS", "x small": "XS", "extra small": "XS",
    "small": "S",
    "medium": "M",
    "large": "L",
    "x-large": "XL", "x large": "XL", "extra large": "XL",
    "xx-large": "XXL", "xx large": "XXL", "extra extra large": "XXL",
}


_SIZE_WORD_RE = re.compile(
    r"(?:,\s*|\bregular\s+)("
    + "|".join(sorted((re.escape(k) for k in _SIZE_WORD_MAP), key=len, reverse=True))
    + r")\s*\)?\s*$",
    re.IGNORECASE,
)

_SPLIT_RE = re.compile(r"\s*[,/|;]\s*")


_DEPARTMENT_NORMALIZE: dict[str, str] = {
    "womens": "women", "women": "women",
    "mens": "men", "men": "men",
    "girls": "girls", "boys": "boys",
    "unisexadult": "unisex", "unisexchild": "unisex", "unisexbaby": "unisex",
    "unisex": "unisex",
    "babygirls": "baby", "babyboys": "baby", "baby": "baby",
}


def _as_list_of_str(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


def _normalize_details(raw_details: Any) -> dict[str, str]:
    if not isinstance(raw_details, dict):
        return {}
    return {
        str(k).strip().lower(): str(v).strip()
        for k, v in raw_details.items()
        if v is not None and str(v).strip()
    }


_PRICE_NUMBER_RE = re.compile(r"\d+\.\d+|\d+")


def _extract_price(row: dict[str, Any]) -> float | None:
    raw = row.get("price")
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    match = _PRICE_NUMBER_RE.search(str(raw))
    return float(match.group()) if match else None


_DEPARTMENT_CATEGORY_WORDS: set[str] = {
    "women", "men", "boys", "girls", "unisex", "baby",
}


def _extract_department(details: dict[str, str], categories: list[str]) -> str | None:
    raw = details.get("department")
    if raw:
        key = re.sub(r"[\s\-]", "", raw.lower())
        return _DEPARTMENT_NORMALIZE.get(key, raw.strip().lower())

    for category in categories:
        key = category.strip().lower()
        if key in _DEPARTMENT_CATEGORY_WORDS:
            return key
    return None


def _scan_words(text: str, vocab: list[str]) -> list[str]:
    text_lower = text.lower()
    matches: list[tuple[int, str]] = []
    for w in vocab:
        m = re.search(rf"\b{re.escape(w)}\b", text_lower)
        if m:
            matches.append((m.start(), w))
    matches.sort(key=lambda pair: pair[0])
    return [w for _, w in matches]


def _extract_colors(details: dict[str, str], scan_text: str) -> list[str]:
    detail_val = details.get("color")
    if detail_val:
        return [v for v in _SPLIT_RE.split(detail_val) if v]
    return _scan_words(scan_text, _COLOR_WORDS)


def _extract_material(details: dict[str, str], scan_text: str) -> str | None:
    detail_val = details.get("material")
    if detail_val:
        parts = [v for v in _SPLIT_RE.split(detail_val) if v]
        return parts[0] if parts else None
    found = _scan_words(scan_text, _MATERIAL_WORDS)
    return found[0] if found else None


def _extract_sizes(details: dict[str, str], title: str) -> list[str]:
    detail_val = details.get("size")
    if detail_val:
        return [v for v in _SPLIT_RE.split(detail_val) if v]

    letters = [s.upper() for s in _SIZE_LETTER_RE.findall(title)]
    numerics = _SIZE_NUMERIC_RE.findall(title)
    word_match = _SIZE_WORD_RE.search(title)
    words = [_SIZE_WORD_MAP[word_match.group(1).lower()]] if word_match else []

    seen: set[str] = set()
    result: list[str] = []
    for s in [*letters, *numerics, *words]:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _build_search_text(row: dict[str, Any], title: str) -> str:
    features = _as_list_of_str(row.get("features"))
    description = _as_list_of_str(row.get("description"))
    categories = _as_list_of_str(row.get("categories"))
    return " ".join([title, *features, *description, *categories]).strip()


def _build_attribute_scan_text(row: dict[str, Any], title: str) -> str:
    features = _as_list_of_str(row.get("features"))
    return " ".join([title, *features]).strip()


def dense_text(row: dict[str, Any]) -> str:
    """Text embedded for dense retrieval. Identical to `Candidate.search_text` so
    that `scripts/build_artifacts.py` precomputes exactly what
    `DenseIndex.build()` would encode at startup."""
    title = str(row.get("title") or "").strip()
    return _build_search_text(row, title)


def normalize_row(row: dict[str, Any]) -> Candidate | None:
    parent_asin = row.get("parent_asin")
    if not parent_asin:
        log.warning("Skipping row with no parent_asin: %r", row.get("title", "<no title>"))
        return None

    title = str(row.get("title") or "").strip()
    details = _normalize_details(row.get("details"))
    categories = _as_list_of_str(row.get("categories"))
    search_text = _build_search_text(row, title)
    attribute_scan_text = _build_attribute_scan_text(row, title)
    brand = row.get("store")

    return Candidate(
        parent_asin=str(parent_asin),
        title=title,
        brand=str(brand).strip() if brand else None,
        categories=categories,
        colors=_extract_colors(details, attribute_scan_text),
        sizes=_extract_sizes(details, title),
        material=_extract_material(details, attribute_scan_text),
        department=_extract_department(details, categories),
        price=_extract_price(row),
        average_rating=float(row.get("average_rating") or 0.0),
        rating_number=int(row.get("rating_number") or 0),
        search_text=search_text,
    )


def iter_catalog(path: str | Path) -> Iterator[Candidate]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed JSON at line %d", line_no)
                continue
            candidate = normalize_row(row)
            if candidate is not None:
                yield candidate


def load_catalog(path: str | Path) -> dict[str, Candidate]:
    return {c.parent_asin: c for c in iter_catalog(path)}


def _diagnose(path: str, sample: int) -> None:
    # confirm the coverage of the attributes extracted
    total = 0
    hits = {"color": 0, "size": 0, "material": 0, "price": 0, "brand": 0, "department": 0}
    shown = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total += 1
            candidate = normalize_row(row)
            if candidate is None:
                continue
            hits["color"] += bool(candidate.colors)
            hits["size"] += bool(candidate.sizes)
            hits["material"] += bool(candidate.material)
            hits["price"] += candidate.price is not None
            hits["brand"] += bool(candidate.brand)
            hits["department"] += bool(candidate.department)

            if shown < sample:
                print("--- raw keys:", sorted(row.keys()))
                print("--- normalized:", candidate)
                shown += 1

    print(f"\nScanned {total} rows.")
    for attr, count in hits.items():
        pct = (count / total * 100) if total else 0.0
        print(f"  {attr:10s}: {count:6d}/{total} ({pct:.1f}%) resolved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog_path")
    parser.add_argument("--sample", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    _diagnose(args.catalog_path, args.sample)
