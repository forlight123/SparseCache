import torch

from experiments.lossless_pd.lmcache_pd.benchmark_pd_packets import (
    expected_output_ids,
    select_packet_entries,
    summarize,
)
from experiments.lossless_pd.lmcache_pd.compare_online_modes import sandwich_summary


def test_expected_output_ids_respects_total_api_horizon():
    packet = {
        "seed": torch.tensor([[11]]),
        "reference": torch.tensor([13, 17, 19, 23]),
    }
    assert expected_output_ids(packet, 3) == [11, 13, 17]


def test_packet_summary_counts_exact_ids():
    rows = [
        {
            "ordinal": 0,
            "outputs_equal": True,
            "pd": {"headers_ms": 1.0, "ttft_ms": 2.0, "total_ms": 4.0},
        },
        {
            "ordinal": 1,
            "outputs_equal": False,
            "pd": {"headers_ms": 3.0, "ttft_ms": 4.0, "total_ms": 8.0},
        },
    ]
    result = summarize(rows)
    assert result["outputs_equal_reference"] == 1
    assert result["mismatch_ordinals"] == [1]
    assert result["pd"]["total_ms"] == 6.0


def test_sandwich_supports_reference_only_packet_runs():
    def row(total: float):
        return {
            "record_id": "r0",
            "pd": {"text": "unstable", "token_ids": [1, 2], "ttft_ms": 5.0, "total_ms": total},
            "outputs_equal": True,
        }

    result = sandwich_summary([row(10.0)], [row(8.0)], [row(12.0)])
    assert result["all_three_pd_outputs_equal"] == 1
    assert result["all_three_equal_reference"] == 1
    assert result["inject_minus_sandwich_observe_total_ms"]["mean_ms"] == -3.0
    assert "sandwich_difference_in_differences_total_ms" not in result


def test_select_packet_entries_preserves_explicit_order():
    entries = [{"index": 25}, {"index": 57}, {"index": 61}]
    assert select_packet_entries(entries, 1, "61,25") == [entries[2], entries[0]]


def test_select_packet_entries_rejects_missing_or_duplicates():
    entries = [{"index": 25}]
    for requested in ("25,25", "99"):
        try:
            select_packet_entries(entries, 1, requested)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {requested}")
