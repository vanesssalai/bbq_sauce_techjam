"""Score copilot.agent.Agent on the public set using the organizer's evaluator.

    python scripts/evaluate.py                 # one run, prints table + summary
    python scripts/evaluate.py --seeds 5       # repeat, report mean/stdev
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from copilot.agent import Agent

COMPOSITE_KEYS = ("hit_rate_at_10", "mrr", "mttc", "efficiency", "recommended_technical_score")


def _baseline_delta(summary: dict, baseline_path: Path) -> dict | None:
    if not baseline_path.exists():
        return None
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    delta = {
        key: round(summary[key] - baseline[key], 6)
        for key in ("hit_rate_at_10", "mrr", "mttc", "efficiency")
        if key in summary and key in baseline
    }
    if "recommended_technical_score" in summary and "technical_score" in baseline:
        delta["technical_score"] = round(
            summary["recommended_technical_score"] - baseline["technical_score"], 6
        )
    return delta


def _scenario_table(scenario_metrics: dict) -> str:
    header = f"{'scenario':<16}{'n':>5}{'hit@10':>10}{'mrr':>10}{'mttc':>10}"
    rows = [header, "-" * len(header)]
    for name, m in scenario_metrics.items():
        mttc = float("nan") if m["mttc"] is None else m["mttc"]
        rows.append(
            f"{name:<16}{m['sample_count']:>5}{m['hit_rate_at_10']:>10.4f}"
            f"{m['mrr']:>10.4f}{mttc:>10.4f}"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalog", default=str(REPO_ROOT / "data" / "catalog.jsonl"))
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / "public_set.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "results.json"))
    parser.add_argument("--baseline", default=str(REPO_ROOT / "docs" / "baseline_results.json"))
    parser.add_argument(
        "--seeds", type=int, default=1,
        help="Run the evaluation N times; report mean/stdev of the composite metrics "
             "(meaningful once the agent has nondeterministic LLM behavior).",
    )
    # Reserved; wired into copilot.agent once the retrieval/ranking pipeline lands.
    parser.add_argument("--no-prf", action="store_true", help="(reserved) disable pseudo-relevance feedback")
    parser.add_argument("--no-dense", action="store_true", help="(reserved) disable dense retrieval")
    parser.add_argument("--no-tournament", action="store_true", help="(reserved) disable cross-encoder rerank")
    args = parser.parse_args()

    reserved = [name for name in ("no_prf", "no_dense", "no_tournament") if getattr(args, name)]
    if reserved:
        print(f"warning: {reserved} not wired up yet; ignored", file=sys.stderr)

    samples = load_jsonl(args.dataset)
    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    runs = [
        evaluate(agent, samples, catalog_ids, categories, products)
        for _ in range(max(1, args.seeds))
    ]
    result = runs[-1]
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    summary = {key: value for key, value in result.items() if key != "sessions"}
    if len(runs) > 1:
        summary["seed_runs"] = {
            key: {
                "mean": round(statistics.fmean(run[key] for run in runs), 6),
                "stdev": round(statistics.pstdev(run[key] for run in runs), 6),
            }
            for key in COMPOSITE_KEYS
            if key in result
        }
    delta = _baseline_delta(summary, Path(args.baseline))
    if delta is not None:
        summary["delta_vs_baseline"] = delta

    print(_scenario_table(result["scenario_metrics"]))
    print()
    print(json.dumps({k: v for k, v in summary.items() if k != "scenario_metrics"}, indent=2))


if __name__ == "__main__":
    main()
