import pytest

from experiments.lossless_pd.lmcache_pd.benchmark import (
    bootstrap_mean_ci,
    select_entries,
    summarize,
)


def test_select_entries_spans_prompt_lengths():
    entries = [
        {"index": index, "prompt_tokens": tokens}
        for index, tokens in enumerate([800, 100, 400, 200, 600])
    ]
    selected = select_entries(entries, 3)
    assert [entry["prompt_tokens"] for entry in selected] == [100, 400, 800]


def test_summarize_uses_paired_latency_deltas():
    rows = [
        {
            "pd": {"headers_ms": 3.0, "ttft_ms": 10.0, "total_ms": 20.0},
            "monolithic": {
                "headers_ms": 1.0,
                "ttft_ms": 7.0,
                "total_ms": 15.0,
            },
            "outputs_equal": True,
        },
        {
            "pd": {"headers_ms": 5.0, "ttft_ms": 8.0, "total_ms": 18.0},
            "monolithic": {
                "headers_ms": 1.0,
                "ttft_ms": 9.0,
                "total_ms": 15.0,
            },
            "outputs_equal": False,
        },
    ]
    result = summarize(rows)
    assert result["pd"]["ttft_ms"] == 9.0
    assert result["monolithic"]["ttft_ms"] == 8.0
    assert result["pd_minus_monolithic_ttft_ms"]["mean_ms"] == 1.0
    assert result["pd_minus_monolithic_ttft_ms"]["pd_faster"] == 1
    assert result["outputs_equal"] == 1


def test_bootstrap_singleton_is_exact():
    assert bootstrap_mean_ci([2.5]) == pytest.approx([2.5, 2.5])
