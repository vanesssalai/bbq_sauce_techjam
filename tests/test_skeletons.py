"""Smoke tests: modules import, contract types exist, fixtures are well formed,
and the wired retrieval->rank contract holds on the fixture pool.

Run with `pytest -q` or directly: `python tests/test_skeletons.py`.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = [
    "copilot.contracts",
    "copilot.models",
    "copilot.retrieval.query",
    "copilot.retrieval.bm25",
    "copilot.retrieval.dense",
    "copilot.retrieval.fusion",
    "copilot.retrieval.prf",
    "copilot.retrieval.retrieve",
    "copilot.ranking",
    "copilot.ranking.rank",
    "copilot.dialog.distill",
]


def test_modules_import():
    for name in MODULES:
        importlib.import_module(name)


def test_new_contract_types_exist():
    from copilot.contracts import Query, RankResult

    assert Query is not None and RankResult is not None


def test_fixture_pool_shape():
    from tests.fixtures import TARGET_ASIN, fused_candidates

    pool = fused_candidates()
    assert [c.fused_rank for c in pool] == [1, 2, 3, 4, 5, 6]
    assert [c.fused_score for c in pool] == sorted((c.fused_score for c in pool), reverse=True)
    assert any(c.parent_asin == TARGET_ASIN for c in pool)

    pool[0].rank_score = 1.0
    assert fused_candidates()[0].rank_score is None  # fresh copy each call


def test_query_contract_is_unified():
    # The two parallel-branch Query shapes were merged into one superset; both
    # the retrieval fields and the rerank fields must be constructible together.
    from copilot.contracts import Query

    q = Query(
        free_text="leather boots", slots={}, hard_slots={}, soft_slots={},
        negations={}, category_anchor="boots", intent_p_buying=0.8,
        track="buying", turn=1,
    )
    assert q.is_empty is False


def test_rank_smoke_on_fixture_pool():
    # rank() is fully implemented now (Track K). Degraded mode (no cross-encoder)
    # must still return a valid RankResult from the fixture pool.
    from copilot.contracts import RankResult
    from copilot.ranking.rank import rank
    from tests.fixtures import fused_candidates, make_fixture_query, make_fixture_session

    res = rank(
        fused_candidates(), make_fixture_query(), make_fixture_session(),
        top_k=3, cross_encoder=None,
    )
    assert isinstance(res, RankResult)
    assert len(res.ranked) == 3
    assert all(c.rank_score is not None for c in res.ranked)
    assert len({c.parent_asin for c in res.ranked}) == 3


def test_profile_calib_importable():
    from copilot.dialog.distill import profile_calib
    from tests.fixtures import fused_candidates

    assert profile_calib(fused_candidates()[0], None) == 0.0


if __name__ == "__main__":
    for _name, _fn in sorted((k, v) for k, v in dict(globals()).items() if k.startswith("test_")):
        _fn()
        print(f"ok  {_name}")
    print("all passed")
