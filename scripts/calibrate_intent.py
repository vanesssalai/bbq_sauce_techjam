from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
)

from copilot.dialog.intent import centroid
from copilot.models import Encoder


def reconstruct_first_messages(public_path: str, catalog_path: str) -> dict[str, list[str]]:
    samples = load_jsonl(public_path)
    _ids, categories, products = catalog_index(catalog_path)
    buckets: dict[str, list[str]] = {"buying": [], "browsing": []}
    for s in samples:
        card, behavior = materialize_hidden_fields(s, products)
        effective = {**s, "intent_card": card, "behavior": behavior}
        target = str(s["ground_truth"]["parent_asin"])
        cat = coarse_category(categories.get(target, []))
        msg = initial_message(effective, cat, set())
        buckets["buying" if s.get("scenario_type") == "buying" else "browsing"].append(msg)
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--public", default="data/public_set.jsonl")
    ap.add_argument("--catalog", default="data/catalog.jsonl")
    ap.add_argument("--out", default="data/intent_anchors.json")
    args = ap.parse_args()

    buckets = reconstruct_first_messages(args.public, args.catalog)
    encoder = Encoder()
    buy_vecs = encoder.encode(buckets["buying"])
    browse_vecs = encoder.encode(buckets["browsing"])

    payload = {
        "model": encoder.name,
        "dim": int(encoder.dim),
        "n_buying": len(buckets["buying"]),
        "n_browsing": len(buckets["browsing"]),
        "buy_vectors": [centroid(buy_vecs)],
        "browse_vectors": [centroid(browse_vecs)],
    }
    Path(args.out).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}  ·  model={encoder.name}  ·  "
        f"{payload['n_buying']} buying / {payload['n_browsing']} browsing turn-1 messages"
    )


if __name__ == "__main__":
    main()