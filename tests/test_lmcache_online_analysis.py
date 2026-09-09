from experiments.lossless_pd.lmcache_pd.analyze_online_draft import summarize
from experiments.lossless_pd.lmcache_pd.compare_online_modes import (
    paired_summary,
    sandwich_summary,
)
from experiments.lossless_pd.lmcache_pd.direct_token_proxy import (
    is_token_id_prompt,
)
from experiments.lossless_pd.lmcache_pd.eval_runtime_layout import (
    runtime_anchor_positions,
)


def draft(request, proposals=(1, 2, 3)):
    return {
        "event": "live_sparse_draft",
        "request_id": request,
        "total_gpu_ms": 2.0,
        "wall_ms": 2.5,
        "actual_fraction": 0.1,
    }


def handoff(request, match, returned=()):
    return {
        "event": "online_draft_handoff",
        "request_id": request,
        "first_proposal_matches": match,
        "returned_proposals": list(returned),
        "draft_ready_lead_ms": 50.0,
    }


def test_online_trace_summary_counts_target_verified_progress():
    rows = [
        {"event": "live_sparse_draft_warmup"},
        draft("a"),
        draft("b"),
        handoff("a", True, (2, 3)),
        handoff("b", False),
        {
            "event": "online_verify_feedback",
            "request_id": "a",
            "accepted_prefix": 2,
        },
    ]
    result = summarize(rows)
    assert result["requests_with_draft"] == result["handoffs"] == 2
    assert result["first_token_match_rate"] == 0.5
    assert result["accepted_prefix_distribution_injected"] == {"2": 1}
    assert result["mean_accepted_prefix_per_request"] == 1.0
    assert result["mean_accepted_injected_suffix_per_request"] == 0.5


def test_online_trace_summary_counts_late_draft_as_zero_progress():
    rows = [
        draft("a"),
        draft("b"),
        handoff("a", True, (2, 3)),
        {
            "event": "online_verify_feedback",
            "request_id": "a",
            "accepted_prefix": 3,
            "accepted_injected_suffix": 2,
        },
    ]

    result = summarize(rows)

    assert result["requests_with_draft"] == 2
    assert result["handoffs"] == 1
    assert result["handoff_rate"] == 0.5
    assert result["mean_accepted_prefix_per_request"] == 1.5
    assert result["mean_accepted_injected_suffix_per_request"] == 1.0


def benchmark_row(record, pd_text, mono_text, pd_total, mono_total):
    return {
        "record_id": record,
        "pd": {"text": pd_text, "ttft_ms": pd_total / 2, "total_ms": pd_total},
        "monolithic": {
            "text": mono_text,
            "ttft_ms": mono_total / 2,
            "total_ms": mono_total,
        },
        "outputs_equal": pd_text == mono_text,
    }


def test_online_mode_comparison_preserves_outputs_and_adjusts_run_drift():
    observe = [benchmark_row("a", "x", "x", 20, 10)]
    inject = [benchmark_row("a", "x", "x", 19, 8)]
    result = paired_summary(observe, inject)
    assert result["inject_equals_observe_pd"] == 1
    assert result["inject_minus_observe_total_ms"]["mean_ms"] == -1
    assert result["difference_in_differences_total_ms"]["mean_ms"] == 1
    assert result["same_monolithic_mismatch_set"] is True


def test_sandwich_comparison_uses_mean_of_surrounding_baselines():
    before = [benchmark_row("a", "x", "x", 20, 10)]
    inject = [benchmark_row("a", "x", "x", 16, 10)]
    after = [benchmark_row("a", "x", "x", 18, 10)]
    result = sandwich_summary(before, inject, after)
    assert result["all_three_pd_outputs_equal"] == 1
    assert result["all_three_equal_monolithic"] == 1
    assert result["monolithic_mismatch_ordinals"] == []
    assert result["inject_minus_sandwich_observe_total_ms"]["mean_ms"] == -3
    assert result["observe_after_minus_before_total_ms"]["mean_ms"] == -2


def test_sandwich_comparison_reports_monolithic_mismatch():
    before = [benchmark_row("a", "x", "x", 20, 10)]
    inject = [benchmark_row("a", "x", "y", 16, 10)]
    after = [benchmark_row("a", "x", "x", 18, 10)]
    result = sandwich_summary(before, inject, after)
    assert result["all_three_pd_outputs_equal"] == 1
    assert result["all_three_equal_monolithic"] == 0
    assert result["monolithic_mismatch_ordinals"] == [0]


def test_runtime_layout_matches_protected_uniform_lmcache_chunks():
    positions = runtime_anchor_positions(
        7800, 256, 0.1, "protected_uniform", device="cpu"
    )
    assert positions.numel() == 888
    assert positions[:3].tolist() == [0, 1, 2]
    assert positions[255:258].tolist() == [255, 2560, 2561]
    assert positions[-3:].tolist() == [7797, 7798, 7799]


def test_direct_token_proxy_distinguishes_tokens_from_text_and_batches():
    assert is_token_id_prompt([1, 2, 3])
    assert not is_token_id_prompt("1 2 3")
    assert not is_token_id_prompt([[1, 2, 3]])
    assert not is_token_id_prompt([True, 2])
    assert not is_token_id_prompt([])
