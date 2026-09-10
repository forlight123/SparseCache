import pytest

from experiments.lossless_pd.screen_small_model_draft import (
    accepted_prefix,
    percentile,
    summarize,
)


def test_accepted_prefix_stops_on_mismatch_or_target_eos():
    assert accepted_prefix([1, 2, 9], [1, 2, 3], set()) == 2
    assert accepted_prefix([1, 2, 9], [1, 2, 3], {2}) == 2
    assert accepted_prefix([9], [1], set()) == 0


def test_percentile_interpolates_and_rejects_empty_input():
    assert percentile([1.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError):
        percentile([], 0.95)


def test_summary_keeps_acceptance_and_cost_units_separate():
    rows = [
        {
            "accepted": 0,
            "prefill_ms": 10.0,
            "draft_decode_ms": 2.0,
            "total_draft_ms": 12.0,
            "draft_tokens": 4,
        },
        {
            "accepted": 4,
            "prefill_ms": 14.0,
            "draft_decode_ms": 4.0,
            "total_draft_ms": 18.0,
            "draft_tokens": 4,
        },
    ]
    result = summarize(rows)

    assert result["mean_accepted_prefix"] == 2.0
    assert result["zero_acceptance_rate"] == 0.5
    assert result["full_acceptance_rate"] == 0.5
    assert result["mean_prefill_ms"] == 12.0
    assert result["mean_draft_decode_ms"] == 3.0
    assert result["mean_total_draft_ms"] == 15.0


def test_summary_reports_concurrent_seed_branch_hit_rate():
    rows = [
        {
            "accepted": accepted,
            "prefill_ms": None,
            "draft_decode_ms": None,
            "total_draft_ms": 10.0,
            "draft_tokens": 4,
            "seed_branch_hit": hit,
        }
        for accepted, hit in ((4, True), (0, False))
    ]

    assert summarize(rows)["seed_branch_hit_rate"] == 0.5
