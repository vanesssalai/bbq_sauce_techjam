# Retrieval Handoff — what the dialog layer hands you

Scope: everything from `build_query(...)` to a ranked `list[Candidate]`. The
dialog side (NLU → state → query) is done and wired; you own retrieve → fuse →
rerank. Baseline to beat, measured on `python -m evaluator.local_evaluator`
(200 public sessions): **HitRate@10 0.93 · MRR 0.554 · MTTC 4.59 ·
TechnicalScore 0.760**.

---

## 1. The seam

In [`copilot/agent.py`](copilot/agent.py) `_respond()`, right after `apply_turn`:

```python
query = build_query(state, parsed)                 # <-- dialog output (Section 2)
pool_asins = self._bm25_pool(query.free_text or user_message,
                             max(top_k * _POOL_MULTIPLIER, _POOL_MIN))  # 400
pool = [replace(self._catalog[a]) for a in pool_asins if a in self._catalog]

survivors, relaxation = apply_filters_with_relaxation(pool, query.hard_slots, query.negations)
ranked = pool                                      # <-- YOU REPLACE THIS LINE

unseen  = [c for c in ranked if c.parent_asin not in state.shown_asins]
ordered = unseen if len(unseen) >= top_k else unseen + [c for c in ranked if c.parent_asin in state.shown_asins]
recommendations = [{"parent_asin": c.parent_asin} for c in ordered[:top_k]]
```

**Your job:** produce `ranked: list[Candidate]`, best→worst, deduped by
`parent_asin`, length ≥ `top_k` when possible. The agent does the
already-shown exclusion and the top-10 slice after you. Everything downstream of
`ranked =` stays as is.

**Inputs available to you at that point**

| what | type | notes |
|---|---|---|
| `query` | `Query` | Section 2 — the structured hand-off |
| `self._catalog` | `dict[str, Candidate]` | all 50k, attributes extracted, keyed by `parent_asin` |
| `self.connection` | `sqlite3.Connection` | in-memory FTS5 table `products` (Section 4) |
| `self._bm25_pool(text, limit)` | `-> list[str]` | the current lexical channel, ranked `parent_asin`s |
| `pool` | `list[Candidate]` | fresh per-turn **copies** — safe to write score fields onto |
| `survivors`, `relaxation` | see Section 5 | hard-filter output; currently used for dialogue only |
| `top_k` | `int` | always 10 |
| `state` | `SessionState` | `state.shown_asins`, `state.current_track`, `state.user_profile`, … if you want more signal |

Suggested shape (yours to decide): a `Retriever` built once in `Agent.__init__`
(needs `self.connection`, `self._catalog`, an artifacts dir) with
`search(query: Query, *, limit: int) -> list[Candidate]`. Keep it a plain
function/class returning `list[Candidate]`; don't change `respond()`'s outer
contract.

---

## 2. `Query` — [`copilot/retrieval/query.py`](copilot/retrieval/query.py)

```python
@dataclass
class Query:
    hard_slots: dict[str, Slot]        # user-STATED, filterable constraints
    soft_slots: dict[str, Slot]        # colour / material / brand — boosts only
    negations:  dict[str, list[str]]   # attr -> excluded values
    free_text:  str                    # the string for BM25 + dense
    category_anchor: str               # verbatim turn-1 category phrase
    intent_p_buying: float             # 0..1 EMA, buying vs browsing
```

`build_query(session, parsed=None) -> Query`. Call it after `apply_turn`.
`parsed` is optional; it only adds `rewritten_query` (LLM, currently always
absent) and this turn's `soft_tags` to `free_text`.

### `free_text`
Assembled and token-deduped in this priority order:
1. `parsed.rewritten_query` (if present)
2. `category_anchor`
3. open-ended slots: `category`, `department`, `style`, `use_case`
4. every `session.disclosed_phrases` entry (accumulated, deduped, capped 16)
5. soft-slot words: `color`, `material`, `brand`
6. `parsed.soft_tags`

De-dup is token-level: a part whose content words are all already present is
dropped (`"earrings"` disappears under anchor `"Earrings Hoop"`; `"silver"`
under `"sterling silver"`). **Price is deliberately not in `free_text`** — it's
in `hard_slots`. Falls back to the latest raw user message if everything else is
empty. This is the string to feed both the lexical and the dense encoder.

### `hard_slots`  (keys ⊆ `{category, department, size, price_min, price_max}`)
Only slots the user **stated outright** — `source ∈ {explicit,
clarification_answer}` and effective confidence ≥ 0.5. Inferred / LLM /
semantic-prototype guesses are excluded on purpose, so a bad guess can't filter
the target out. Values are `Slot` objects → use `slot.value`.
`filters.apply_filters` already consumes this dict directly.

### `soft_slots`  (keys ⊆ `{color, material, brand}`)
`Slot` objects. Not filtered on (a `parent_asin` spans colours/SKUs). Use as
rerank boosts — reward a candidate whose `.colors` / `.material` / `.brand`
matches. Their words are also already folded into `free_text`.

### `negations`  — `{attr: [values]}`
From explicit "not X / no X / other than X". `filters._violates_negation` only
acts on `color` / `material` / `brand`. Honour these as a hard exclusion or a
strong penalty; never surface a negated value.

### `category_anchor`
The category phrase parsed verbatim from the turn-1 message (the simulator
always opens `"I'm looking for {category}…"`; `{category}` is built from the
target's own `categories` list, so it's a high-signal anchor). Good as a
dedicated rerank feature: candidate whose `.categories` contains / is contained
by the anchor → boost.

### `intent_p_buying`
Session-level EMA. High (→1) = buyer with firm intent → lean lexical / precision.
Low (→0) = browser → lean dense / diversify (MMR). Use it plus
`len(query.hard_slots)` for the fusion blend weight.

---

## 3. `Candidate` — [`copilot/contracts.py`](copilot/contracts.py)

```python
@dataclass
class Candidate:
    parent_asin: str
    title: str
    brand: str | None            # from catalog "store"
    categories: list[str]
    colors: list[str]
    sizes: list[str]
    material: str | None
    department: str | None
    price: float | None
    average_rating: float        # weak popularity prior only — NOT relevance
    rating_number: int
    search_text: str             # title + features + description + categories
    # --- you fill these ---
    bm25_score: float = 0.0
    filter_match: bool = False   # set by filters.apply_filters
    dense_score: float = 0.0
    fused_score: float | None = None
    fused_rank: int | None = None
    rank_score: float | None = None
```

`catalog.normalize_row` populates everything above the line. The score fields
are yours. `pool` candidates are per-turn `dataclasses.replace` copies, so
writing scores onto them is safe and can't leak across sessions/turns — **keep
copying** if you restructure.

**Coverage on the real 50k** (from `python -m copilot.catalog data/catalog.jsonl`):
category ~100%, department ~95%, size ~15%, price ~21%, brand/colour partial.
Missing values must never make a product impossible — `filters` already passes
every `None`; do the same in rerank features.

---

## 4. FTS5 index

Built in `Agent._build_index`. Virtual table `products`, columns in this order:
`parent_asin(UNINDEXED), title, categories, features, details, store,
description`, tokenizer `unicode61 remove_diacritics 2`.

Current lexical query (`_bm25_pool`): dedup terms from `free_text`, cap 40,
`"t1" OR "t2" OR …`, ordered by
`bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0)` (title 6 · categories 4 ·
features/details 2.5 · store 1.5 · description 1.0), `LIMIT`.

Handoff §6 wants this extracted to `retrieval/bm25.py` as
`Bm25Index.search(query_text, allowed_ids, limit)`. Field weights are a tuning
knob — retune against `evaluator/local_evaluator`, don't agonise.

---

## 5. Filters — [`copilot/retrieval/filters.py`](copilot/retrieval/filters.py)

- `apply_filters(cands, slots, negated_values) -> list[Candidate]` — keeps
  `HARD_FILTER_ATTRS` only, passes every missing value, sets `c.filter_match`.
- `suggest_relaxation(cands, slots) -> (attr, new_value|None) | None` — which
  single constraint to loosen, with a concrete relaxed number for price.
- `apply_filters_with_relaxation(...) -> (survivors, relaxation|None)` — one call.

**Current state:** `_respond` calls this but sets `ranked = pool` — i.e. the
hard filter is **computed but not applied to ranking**. Cutting or reordering on
today's NLU slots measured net-negative on the public set (–0.01 to –0.02
TechScore) because there's no reranker yet to exploit the smaller set.

**When your cross-encoder lands:** change `ranked = pool` →
`ranked = survivors or pool`, and/or feed `c.filter_match` (and per-attr match
of `soft_slots` / `category_anchor`) as rerank features. Re-measure; keep
whichever wins. Leave the `apply_filters_with_relaxation` call in place — the
`relaxation` value drives a dialogue branch (`_respond` offers to loosen a
constraint when `survivors` is empty; the user's yes/no is resolved by
`state_machine._resolve_pending_relaxation`).

---

## 6. Response contract (don't break this)

`respond()` must return:
```python
{"message": str, "ask_attribute": str | None,
 "recommendations": [{"parent_asin": str}, ...],
 "usage": {"prompt_tokens": int, "completion_tokens": int}}
```
The evaluator takes the first 10 valid, unique, in-catalog `parent_asin`s per
turn; a hit records `(rank, turn)` and the session stops. Any exception →
miss for that turn (there's a `try/except` in `respond()` that falls back to
`self._search(user_message, top_k)` — that bare path must keep working).

If you set `c.rank_score`, `TurnResult.to_response` can surface it as `"score"`;
`_respond` currently emits `parent_asin` only. Optional.

---

## 7. Build order (handoff §6)

1. **`retrieval/bm25.py`** — lift `_bm25_pool` out as-is. Zero score change,
   clean seam. (+ optional trigram channel, FDCD-TF-IDF category weighting — §6,
   both flag-gated.)
2. **`retrieval/dense.py`** — bi-encoder, **precomputed** matrix
   (`data/dense_embeddings.npy`, float16, row i ↔ catalog line i) +
   `scripts/build_artifacts.py` + startup fallback if the artifact is missing /
   `catalog_sha256` mismatches. `bge-small-en-v1.5` or `all-MiniLM-L6-v2`.
   Reuse this one encoder instance for `dialog/semantic_slots.py` and
   `dialog/intent.py` too (they're dormant until an encoder is passed).
3. **`retrieval/fusion.py`** — weighted RRF over `[lexical, dense]`, `k≈60`,
   blend weight from `query.intent_p_buying` + `len(query.hard_slots)`.
4. **`ranking/rank.py`** — `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top
   ~200. **This is where the score moves** and where `hard_slots` / `negations`
   / `soft_slots` / `category_anchor` / `average_rating` become rerank
   features. Then flip the filter on (Section 5).
5. `retrieval/prf.py` (Rocchio, flag-gated), MMR for the browsing track — last.

Add `requirements.txt` when `dense.py` introduces real deps.

---

## 8. Gotchas

- **Slots carry NLU noise.** `hard_slots` is source-gated so it's cleaner, but
  `free_text` still can contain mistakes (`"nothing over $15"` parsed as
  `price_min`; short clarification replies bound as raw `color` values; phantom
  `style`/`use_case` from the semantic layer). Don't treat any slot as ground
  truth; the lexical+dense+rerank stack should be robust to a wrong term.
- **Startup ≈ 27 s**, almost all in `catalog.normalize_row` (~70 `re.search`
  per row × 50k). One-time per `Agent()`, amortised over the eval, but it grows
  once you load a dense model. Worth precompiling `catalog._scan_words`' vocab
  into single alternation regexes before you're iterating hard.
- **`_bm25_pool` returns `[]`** when `free_text` has no usable terms — handle
  an empty `pool`.
- **`data/intent_anchors.json`** exists (untracked); `dense_embeddings.npy` and
  friends need `scripts/build_artifacts.py` (not built).
- Per handoff §7: only catalog-wide, query-independent precompute is allowed
  (targets are hidden); private eval uses the same catalog records, so the
  artifact is valid there too.

---

## 9. Quick check

```bash
python -m copilot.retrieval.query                 # build_query smoke demo
python -m evaluator.local_evaluator --output /tmp/r.json   # full 200-session score
```
