import pytest

from experiments.lossless_pd.lmcache_pd.compare_pd_outputs import compare


def result(rows):
    return {
        "rows": [
            {
                "record_id": record_id,
                "pd": {"text": pd},
                "outputs_equal": pd == mono,
            }
            for record_id, pd, mono in rows
        ]
    }


def test_comparison_uses_full_pd_as_authoritative_reference():
    progressive = result([("a", "x", "m"), ("b", "y", "y")])
    full = result([("a", "x", "m"), ("b", "y", "y")])
    summary = compare(progressive, full)
    assert summary["progressive_equals_full_pd"] == 2
    assert summary["same_monolithic_mismatch_set"] is True
    assert summary["progressive_monolithic_mismatch_ordinals"] == [0]


def test_comparison_rejects_different_workloads():
    with pytest.raises(ValueError):
        compare(result([("a", "x", "x")]), result([("b", "x", "x")]))
