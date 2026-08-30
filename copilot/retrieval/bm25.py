"""Lexical (BM25) retrieval over the frozen catalog via SQLite FTS5.

Ported from the starter agent's in-memory index. Built at startup (~2-4 s for
50k rows), no on-disk artifact. `search()` returns `[(parent_asin, score)]`
best-first with `score = -bm25(...)` (positive, higher = better); fusion uses
rank position, not the scale.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}

# fts5 column order below; index 0 (parent_asin) is UNINDEXED. features/details
# are weighted heaviest because the simulator discloses near-verbatim
# features/details substrings.
_COLUMNS = ("parent_asin", "title", "features", "details", "categories", "store", "description")
_BM25_WEIGHTS = (0.0, 3.0, 6.0, 5.0, 2.0, 1.5, 1.0)
_FETCH_LIMIT = 800


def _terms(text: str) -> list[str]:
    return [
        tok.lower()
        for tok in _TOKEN_RE.findall(text)
        if len(tok) > 1 and tok.lower() not in _STOPWORDS
    ]


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    return str(value)


class Bm25Index:
    def __init__(self, catalog_path: str | Path) -> None:
        self.catalog_path = Path(catalog_path)
        self._conn = sqlite3.connect(":memory:")
        self._build()

    def _build(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, features, details, categories, store, description, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        batch: list[tuple] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                batch.append(tuple(_cell(row.get(col)) if col != "parent_asin"
                                   else str(row["parent_asin"]) for col in _COLUMNS))
                if len(batch) >= 1000:
                    cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
                    batch.clear()
        if batch:
            cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?,?)", batch)
        self._conn.commit()

    def search(
        self,
        query_text: str,
        allowed_ids: set[str] | None = None,
        limit: int = 400,
    ) -> list[tuple[str, float]]:
        terms = list(dict.fromkeys(_terms(query_text)))
        if not terms:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in terms)
        weight_args = ", ".join(str(w) for w in _BM25_WEIGHTS)
        rows = self._conn.execute(
            f"SELECT parent_asin, bm25(products, {weight_args}) AS b "
            f"FROM products WHERE products MATCH ? ORDER BY b LIMIT {_FETCH_LIMIT}",
            (match_expr,),
        ).fetchall()

        hits: list[tuple[str, float]] = []
        for parent_asin, raw in rows:
            if allowed_ids is not None and parent_asin not in allowed_ids:
                continue
            hits.append((str(parent_asin), -float(raw)))
            if len(hits) >= limit:
                break
        return hits
