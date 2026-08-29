from pathlib import Path

import pytest

pytest.importorskip("httpx")

from experiments.benchmark_progressive_pd_scheduling_live import (
    _conditions,
    _parse_sidecars,
    _prefill_group,
)


def test_scheduling_conditions_rotate_complete_cross_product() -> None:
    first = _conditions(
        ("sequential", "bm25"), ("fixed_s1", "continuous"), shift=0
    )
    second = _conditions(
        ("sequential", "bm25"), ("fixed_s1", "continuous"), shift=1
    )

    assert set(first) == {
        ("sequential", "fixed_s1"),
        ("sequential", "continuous"),
        ("bm25", "fixed_s1"),
        ("bm25", "continuous"),
    }
    assert second == first[1:] + first[:1]


def test_scheduling_sidecar_parser_is_fail_closed() -> None:
    assert _parse_sidecars(["sequential=a.jsonl", "bm25=b.jsonl"]) == {
        "sequential": Path("a.jsonl"),
        "bm25": Path("b.jsonl"),
    }
    with pytest.raises(ValueError, match="unique"):
        _parse_sidecars(["bm25=a.jsonl", "bm25=b.jsonl"])
    with pytest.raises(ValueError, match="at least two"):
        _parse_sidecars(["bm25=a.jsonl"])


def test_prefill_group_is_shared_across_all_request_conditions() -> None:
    bm25_fixed = _prefill_group("nonce", 3, 17)
    bm25_continuous = _prefill_group("nonce", 3, 17)
    oracle = _prefill_group("nonce", 3, 17)

    assert bm25_fixed == bm25_continuous
    assert bm25_fixed == oracle
    assert bm25_fixed != _prefill_group("nonce", 4, 17)
    assert bm25_fixed != _prefill_group("nonce", 3, 18)
