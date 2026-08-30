# TechJam Conversational Shopping Agent

A multi-turn shopping assistant for the TechJam Conversational E-Commerce Search
Challenge. Each session it receives an anonymized preference profile and a short
customer message, then has **10 turns** to surface the customer's hidden target
product in a Top-10 list, asking clarifying questions along the way.

## Project Overview

The pipeline, per turn:

```
customer message
  └─ dialog/nlu.py            rule-based slot + intent + disclosed-phrase extraction
  └─ dialog/state_machine.py  merge/decay slots, accumulate disclosed phrases, track pivots
  └─ retrieval/query.py       assemble one free_text string  (anchor + slots + disclosed phrases)
  └─ retrieval/bm25.py        SQLite FTS5 BM25 over title/features/details/categories/store/description
  └─ retrieval/retrieve.py    hydrate ~400 candidates (lexical only by default)
  └─ ranking/rank.py          cross-encoder RRF-blend rerank of the top 30
  └─ agent.py _order + _dialogue  dedupe, unseen-first, Top-10, pick the next question
```

| choice | why |
| --- | --- |
| **Lexical (BM25) is the backbone**, dense retrieval **off** by default | the simulator discloses near-verbatim catalog `features`/`details` text, so exact term match wins; the bi-encoder adds noise (MRR −0.08, buying HR −0.08). |
| **Ask up to 6 clarifying questions** (was 2) | the simulated customer volunteers almost nothing unprompted; every answer dumps near-verbatim feature text into the query. Capping at 2 left buying targets stuck at BM25 rank ~150. |
| **Cross-encoder as an RRF *blend*, never the base** | raw `ms-marco-MiniLM` logits on a category-slot query render are miscalibrated; blending its rank with the retrieval rank (`1/(k+r_fused) + 1/(k+r_ce)`) lifts MRR +0.05 without hurting HR. |
| **`rank()` returns a deep list (200), not 30** | `_order` walks down it across turns; a 30-item cap made every session go stale after turn 3. |
| **Embedding semantic-slot resolver off** | it injected `style`/`use_case` guesses at a 0.42 cosine threshold and wrecked the sparse browsing queries (−0.22 Tech). |
| **BM25 field weights: features/details heaviest**, `porter` stemming | matches what the customer discloses once the agent asks enough questions. |

## Setup and Installation

Python **3.13** (3.10+ works; the pins target 3.13).

```bash
pip install -r requirements.txt
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
```

**Catalog** — download `catalog.jsonl.gz` from the repo's GitHub Release, then:

```bash
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

**Models** — the cross-encoder is required (`models/` is git-ignored). Reconstruct
from the pinned revisions:

```bash
python scripts/download_models.py

python scripts/download_models.py --verify   # re-hash against models/SHA256SUMS
```

## Steps to Reproduce the Results

From the repo root, with `data/catalog.jsonl` and `models/` in place:

```bash
python -m evaluator.local_evaluator
```

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Any material or fabric you're set on?",
            "ask_attribute": "material",            # or one of: category color size style brand
                                                     #            budget feature use_case other  / null
            "recommendations": [{"parent_asin": "B000..."}, ...],   # up to 10
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
```

The session ends when the target `parent_asin` appears in the Top 10 or after
turn 10. See `docs/agent_api_contract.json`.

## Data Source

Catalog and sessions are derived from Amazon Reviews 2023 (McAuley Lab, UCSD),
Clothing/Shoes/Jewelry 5-core leave-last-out split. See `DATA_ATTRIBUTION.md`
before using or redistributing the data.
