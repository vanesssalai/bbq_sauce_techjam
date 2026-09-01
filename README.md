# TechJam Conversational Shopping Agent

A conversational shopping agent for **TechJam 2026:
Conversational E-Commerce Search** built so a that it can run it on a
laptop with no paid API and no hosted LLM.

## Project Overview

Over a frozen 50,000-product catalog, each session has one hidden target product.
The customer talks for up to 10 turns; the agent must surface that product in its
Top-10 as early and as highly-ranked as possible, asking clarifying questions
along the way.

Each turn: **NLU** (rule-based slot/intent extraction, no LLM) → **state update**
→ **query build** → **retrieval** (SQLite FTS5 BM25, optional dense channel) →
**RRF fusion** → **rerank** (cross-encoder blended into the rank via RRF, plus
small soft adjustments) → **dialogue** (return the Top-10, ask up to ~6 grounded
clarifying questions). 

**Key features:**

- **Hybrid retrieval**: SQLite FTS5 BM25 lexical search plus an optional dense
  bi-encoder channel, combined with weighted Reciprocal Rank Fusion.
- **Phrase matching**: the near-verbatim constraint phrases the customer
  discloses are matched as exact FTS5 phrases, not just loose terms.
- **Cross-encoder rerank**: a MiniLM reranker blended into the rank via RRF
  (never used as the raw score), with soft adjustments for constraint
  satisfaction, category/price fit, rating prior, and user-profile calibration.
- **Rule-based NLU**: slots, intent, negation, and disclosed-phrase extraction
  with zero LLM calls; deterministic and instant.
- **Multi-turn state machine**: newest-value-wins slot merge, confidence decay,
  intent-pivot handling, and a "no preference" freeze for the boundary case.
- **Adaptive clarifying questions**: up to ~6 grounded follow-ups; each answer
  feeds feature text back into the query.
- **Offline and resilient**: runs fully offline, degrades gracefully to a
  BM25 / popularity fallback if a model fails to load.

**Small-business focus:** dependencies are just `sentence-transformers` +
`torch` + `numpy` + `huggingface-hub`. The two models total ~220 MB and run on
CPU. Every network path has an offline fallback (`HF_HUB_OFFLINE=1`), so once the
models are vendored there are no further calls and no recurring cost.

## Setup and Installation

Python 3.13 (3.10+ works).

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# catalog
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl

# models (vendored once, then no network needed)
python scripts/download_models.py
python scripts/download_models.py --verify
```

Ship the `models/` folder with the submission: it is git-ignored and official
scoring may run offline.

## Steps to Reproduce the Results

From the repo root, with `data/catalog.jsonl` and `models/` in place:

```bash
python -m evaluator.local_evaluator                                          # default config
```

Other variations:
```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m evaluator.local_evaluator   # offline check
COPILOT_NO_CROSS_ENCODER=1 python -m evaluator.local_evaluator               # fast (~4 min)
```

NLU is rule-based and nothing samples, so results are deterministic.

| build | HitRate@10 | MRR | MTTC | Tech |
| --- | --- | --- | --- | --- |
| BM25 starter (`starter/agent.py`) | 0.930 | 0.554 | 4.59 | 0.760 |
| this agent, no cross-encoder | 0.965 | 0.603 | 3.01 | 0.823 |
| this agent, default | 0.965 | 0.626 | 2.99 | **0.831** |

Optional demo UI: 
```bash
pip install streamlit && streamlit run app.py
```

## Limitations & What We'd Improve With More Time

- **`boundary` scenario stuck at HitRate 0.80**: when the customer says "no
  preference", late-turn signal is thin. Next step would be to bind the user profile's
  preference tags into the query on turn 1 instead of only as a late ranking nudge.
- **Cross-encoder cost**: a full CPU eval takes ~20–30 min. A distilled or
  quantised reranker would cut per-turn latency.
- **Clarifying questions use fixed templates**: only `ask_attribute` is scored,
  but real phrasing could be drawn from the retrieved pool ("cotton, polyester,
  or leather?"): still without an LLM.
- **Single-query retrieval**: firing each disclosed constraint as its own BM25
  query and fusing would better reward candidates matching *all* constraints.

## Contributions

| Member | Area |
| --- | --- |
| **Vanessa** | Dialogue / NLU (`dialog/`), agent orchestration (`agent.py`), query wiring, offline model integration, evaluation & tuning |
| **Leonard** | Retrieval stack: `retrieval/bm25.py`, `dense.py`, `fusion.py`, `prf.py`, `retrieve.py`, `filters.py`, `catalog.py`, `models.py`, `scripts/` |
| **Sze Ho** | Reranking: `ranking/rank.py` (cross-encoder rerank, tournament, soft adjustments, MMR), `dialog/distill.py`, rank fixtures |

## Data Source

Derived from Amazon Reviews 2023 (McAuley Lab, UCSD). See `DATA_ATTRIBUTION.md`.
